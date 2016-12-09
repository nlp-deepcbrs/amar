require "lfs";
tds = require "tds";
stringx = require "pl.stringx";
file = require "pl.file";
cjson = require "cjson";

--[[
    Reads the text descriptions associated to each item from a given path.

    Parameters:
        - train_path: folder in which item descriptions are stored
        - extension: extension associated to each file contained in the folder
    Output:
        dictionary structure which contains the following data:
        items: item descriptions
        item2pos: dictionary which maps item ids to position in the dataset
        pos2item: dictionary which maps position in the dataset to item ids
        token2id: dictionary which maps tokens to word identifiers
        max_item_len: maximum number of words in a text description
 ]]
function read_items_data(train_path, extension)
    local items = tds.Hash()
    local item2pos = tds.Hash()
    local pos2item = tds.Hash()
    local token2id = tds.Hash()
    local num_items = 1
    local num_tokens = 1
    local max_item_len = 0
    extension = extension or ".tokens"

    for file in lfs.dir(train_path) do
        if file ~= "." and file ~= ".." then
            local attr = lfs.attributes(train_path .. "/" .. file)

            if attr.mode == "file" then
                local item_id = string.sub(file, 0, string.find(file, extension .. "[^." .. extension .. "]*$") - 1)
                local item_words = tds.Vec()
                item2pos[item_id] = num_items
                pos2item[num_items] = item_id

                for line in io.lines(train_path .. "/" .. file) do
                    for word in string.gmatch(line, "%S+") do
                        if token2id[word] == nil then
                            token2id[word] = num_tokens
                            num_tokens = num_tokens + 1
                        end

                        item_words:insert(token2id[word])
                    end
                end

                if #item_words > max_item_len then
                    max_item_len = #item_words
                end

                items[item_id] = item_words
                num_items = num_items + 1
            end
        end
    end

    return {
        items = items,
        item2pos = item2pos,
        pos2item = pos2item,
        token2id = token2id,
        max_item_len = max_item_len
    }
end

--[[
   Loads genres metadata associated to each item.

   Parameters:
    - genres_filename: name of the file containing genres information for each item in JSON format
    - item2pos: maps item ids to item position in the dataset
 ]]
function load_items_genres(genres_filename, item2pos)
    --[[
        Genres mapping example:
        genre2id = {"Science fiction"=1, "Horror"=2, "Thriller"=3, "Comedy"=4, "Action"=5}

        Padded genres:
        1 3 0
        5 0 0
        1 2 3

        Model input: {item, user, genres} -> torch.Tensor elements in the table
    --]]

    local data = cjson.decode(file.read(genres_filename))
    local genre2id = tds.Hash()
    local id2genre = tds.Hash()
    local genres = tds.Hash()
    local num_genres = 1
    local max_num_genres = 0

    for item_id, item_genres in pairs(data) do
        -- item exists in mapping
        if item2pos[item_id] then
            local item = tonumber(item2pos[item_id])
            local item_mapped_genres = tds.Vec()

            if #item_genres > max_num_genres then
                max_num_genres = #item_genres
            end

            for _, item_genre in pairs(item_genres) do
                if not genre2id[item_genre] then
                    genre2id[item_genre] = num_genres
                    id2genre[num_genres] = item_genre
                    num_genres = num_genres + 1
                end

                item_mapped_genres:insert(genre2id[item_genre])
            end

            genres[item] = item_mapped_genres
        end
    end

    return {
        genres = genres,
        genre2id = genre2id,
        id2genre = id2genre,
        max_num_genres = max_num_genres
    }
end

--[[
    Loads ratings from the specified file in a CSV format with a given delimiter.

    Parameters:
        - ratings_filename: filename of the ratings data in the format (user_id, item_id, rating)
        - item2pos: maps item ids to item positions
        - delimiter: delimiter used by the specified rating file
 ]]
function read_ratings_data(ratings_filename, item2pos, delimiter)
    local delimiter = delimiter or ","
    local data = file.read(ratings_filename)
    local file_lines = stringx.splitlines(data)
    local ratings = torch.Tensor(#file_lines, 3)
    local user2id = tds.Hash()
    local id2user = tds.Hash()
    local num_users = 1

    for i = 1, #file_lines do
        local splitted_line = stringx.split(file_lines[i], delimiter)
        local raw_user = tonumber(splitted_line[1])

        if user2id[raw_user] == nil then
            user2id[raw_user] = num_users
            id2user[num_users] = raw_user
            num_users = num_users + 1
        end

        local user = user2id[raw_user]
        local item = tonumber(item2pos[splitted_line[2]])
        local rating = tonumber(splitted_line[3])

        ratings[i][1] = user
        ratings[i][2] = item
        ratings[i][3] = rating
    end

    return {
        ratings = ratings,
        user2id = user2id,
        id2user = id2user
    }
end

--[[
    Pads item description according to the maximum number of tokens in the item descriptions.
 ]]
function pad_items_data(items_data)
    local data = torch.Tensor(#items_data["items"], items_data["max_item_len"]):zero()

    for item_id, tokens in pairs(items_data["items"]) do
        for i, token in pairs(tokens) do
            data[items_data["item2pos"][item_id]][i] = token
        end
    end

    return data
end

--[[
    Pads item genres according to the maximum number of genres associated to each item
 ]]
function pad_genres_data(genres_data)
    local data = torch.Tensor(#genres_data["genres"], genres_data["max_num_genres"]):zero()

    for item_pos, genres in pairs(genres_data["genres"]) do
        for i, genre in pairs(genres) do
            data[item_pos][i] = genre
        end
    end

    return data
end
