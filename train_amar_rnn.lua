require "optim";
require "cutorch";
require "cunn";
require "cudnn";
file = require "pl.file";
cjson = require "cjson";
require "models/amar.lua"
require "datasets/dataset.lua";
require "datasets/utils.lua";
require "utils/progress.lua";

torch.setdefaulttensortype("torch.FloatTensor")
cutorch.setDevice(1)
torch.manualSeed(12345)
cutorch.manualSeed(12345)

function round(num, idp)
    local mult = 10 ^ (idp or 0)
    return math.floor(num * mult + 0.5) / mult
end

local title = "### AMAR (Ask Me Any Rating) RNN trainer ###"

print(title)

--[[
    JSON configuration file parameters:
        - items: path of item descriptions
        - genres: filename of item genres (optional)
        - models_mapping: dictionary which associates training sets to models
        - rnn_unit: RNN unit identifier used in rnn package
        - optim_method: optimization method identifier used in optim package
        - training_params: parameters of the optimization method
        - batch_size: number of training examples in a batch
        - num_epochs: number of training epochs
        - save_after: save model each save_after epochs
--]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text(title)
cmd:text()
cmd:text("Options:")
cmd:option("-config", "", "Filename of JSON training parameters")
cmd:text()

local params = cmd:parse(arg)

local conf_data = cjson.decode(file.read(params.config))

print("-- Loading items data: "..conf_data["items"])
local items_data = read_items_data(conf_data["items"])

print("-- Padding items data")
items_data["items"] = pad_items_data(items_data)

local genres_data

if conf_data["genres"] then
    print("-- Loading genres data: "..conf_data["genres"])
    genres_data = load_items_genres(conf_data["genres"], items_data["item2pos"])

    print("-- Padding genres data")
    genres_data["genres"] = pad_genres_data(genres_data)
end

local rnn_unit_id = conf_data["rnn_unit"]

local optim_method_id = conf_data["optim_method"]
local optim_method

if optim_method_id == "sgd" then
    optim_method = optim.sgd
elseif optim_method_id == "adadelta" then
    optim_method = optim.adadelta
elseif optim_method_id == "adagrad" then
    optim_method = optim.adagrad
elseif optim_method_id == "adam" then
    optim_method = optim.adam
elseif optim_method_id == "rmsprop" then
    optim_method = optim.rmsprop
else
    print("Invalid training method: "..optim_method_id)
end

local batch_size = conf_data["batch_size"]
local num_epochs = conf_data["num_epochs"]

local coeff_l2 = conf_data["training_params"]["coeff_l2"]

if coeff_l2 then
    print("-- Using L2 regularization using coefficient "..coeff_l2)
end

for train_filename, model_filename in pairs(conf_data["models_mapping"]) do
    local training_params = {}

    for k, v in pairs(conf_data["training_params"]) do
        training_params[k] = v
    end

    print("-- Loading ratings data: "..train_filename)
    local ratings_data = read_ratings_data(train_filename, items_data["item2pos"], "\t")
    local num_examples = ratings_data["ratings"]:size(1)

    print("Dataset stats:")
    print("Number of items:\t"..items_data["items"]:size(1))
    print("Vocabulary size:\t"..#items_data["token2id"])
    print("Number of users:\t"..#ratings_data["user2id"])
    print("Number of ratings:\t"..num_examples)

    print("-- Building model: "..model_filename)
    local model = build_model_amar_rnn_fast(items_data, ratings_data, genres_data, batch_size)
    model = cudnn.convert(model, cudnn)
    model = model:cuda()

    local criterion = nn.BCECriterion()
    criterion = criterion:cuda()

    -- get model parameters
    local params, grad_params = model:getParameters()

    local cost_per_epoch = {}

    print("-- Training model with "..rnn_unit_id.." unit using "..optim_method_id)
    for e = 1, num_epochs do
        -- shuffle and split training examples in batches
        local indices = torch.randperm(num_examples):long():split(batch_size)

        -- remove last element so that all the batches have equal size
        indices[#indices] = nil

        print("==> doing epoch on training data:")
        print("==> online epoch # "..e.." [batchSize = "..batch_size.."]")

        local average_cost = 0

        for t, v in ipairs(indices) do
            -- items positions
            local curr_items_ids_batch = torch.reshape(ratings_data["ratings"]:index(1, v)[{ {}, { 2 } }]:long(), batch_size)

            -- items descriptions
            local curr_items_batch = items_data["items"]:index(1, curr_items_ids_batch):cuda()

            -- users ids
            local curr_users_batch = torch.reshape(ratings_data["ratings"]:index(1, v)[{ {}, { 1 } }]:cuda(), batch_size)

            -- model inputs
            local inputs = { curr_items_batch, curr_users_batch }

            if conf_data["genres"] then
                -- genres ids
                local curr_genres_batch = genres_data["genres"]:index(1, curr_items_ids_batch):cuda()
                table.insert(inputs, curr_genres_batch)
            end

            -- model targets
            local targets = ratings_data["ratings"]:index(1, v)[{ {}, { 3 } }]:cuda()

            -- callback that does a single batch optimization step
            local batch_optimize = function(x)
                -- get new parameters
                if x ~= params then
                    params:copy(x)
                end

                -- reset gradients
                grad_params:zero()

                -- backward propagation
                local outputs = model:forward(inputs)
                local f = criterion:forward(outputs, targets)
                local df_do = criterion:backward(outputs, targets)
                model:backward(inputs, df_do)

                if coeff_l2 then
                    -- L2 regularization
                    f = f + coeff_l2 * torch.norm(params, 2) ^ 2 / 2
                    grad_params:add(params:mul(coeff_l2))
                end

                -- return f and df/dX
                return f, grad_params
            end

            -- optimize on current mini-batch
            local _, fs = optim_method(batch_optimize, params, training_params)

            -- evaluate current loss function value
            local current_cost = fs[1]
            average_cost = average_cost + current_cost

            -- show custom progress bar
            progress(t, #indices, round(current_cost, 2))
        end

        -- evaluate average cost per epoch
        average_cost = round(average_cost / #indices, 4)
        print("Average cost per epoch: "..average_cost)
        table.insert(cost_per_epoch, average_cost)

        if e >= 15 and e % conf_data["save_after"] == 0 then
            print("Saving current model...")
            torch.save(model_filename..".e"..e, model)
        end
    end

    -- save experiment data
    print("Saving experiment data...")
    local experiment_data = {}

    experiment_data["training_params"] = {}

    for k, v in pairs(conf_data["training_params"]) do
        if k ~= "tmp" and k ~= "m" then
            experiment_data["training_params"][k] = v
        end
    end

    experiment_data["optim_method"] = optim_method_id
    experiment_data["batch_size"] = batch_size
    experiment_data["num_epochs"] = num_epochs
    experiment_data["cost_per_epoch"] = cost_per_epoch
    file.write(model_filename..".params", cjson.encode(experiment_data))
end
