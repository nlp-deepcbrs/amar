require "rnn";
require "cunn";
require "cudnn";

function build_model_amar_mean(items_data, ratings_data, genres_data, batch_size)
    local item_embeddings_size = 10
    local user_embeddings_size = 10
    local genre_embeddings_size = 10
    local hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
    local num_tokens = #items_data["token2id"]
    local num_users = #ratings_data["user2id"]
    local lookup_table = nn.LookupTableMaskZero(num_tokens + 1, item_embeddings_size)

    local items_model = nn.Sequential()
        :add(lookup_table)
        :add(nn.Mean(2))

    local users_model = nn.Sequential()
        :add(nn.LookupTable(num_users, user_embeddings_size))

    local full_model = nn.Sequential() -- {item, user}

    local parallel_table = nn.ParallelTable()
        :add(items_model)
        :add(users_model)

    if genres_data then
        print("-- Initializing model for genre features")
        hidden_dense_layer_size = hidden_dense_layer_size + genre_embeddings_size
        local genres_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#genres_data["genre2id"] + 1, genre_embeddings_size))
            :add(nn.Mean(2))

        parallel_table:add(genres_model)
    end

    full_model:add(parallel_table)
        :add(nn.JoinTable(2))
        :add(nn.Linear(hidden_dense_layer_size, 1))
        :add(cudnn.Sigmoid())

    return full_model
end

function build_model_amar_rnn(items_data, ratings_data, genres_data, batch_size, rnn_unit_id)
    local item_embeddings_size = 10
    local user_embeddings_size = 10
    local genre_embeddings_size = 10
    local hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
    local num_tokens = #items_data["token2id"]
    local num_users = #ratings_data["user2id"]
    local lookup_table = nn.LookupTableMaskZero(num_tokens + 1, item_embeddings_size)

    if rnn_unit_id == "LSTM" then
        rnn_unit = nn.LSTM(item_embeddings_size, item_embeddings_size)
    elseif rnn_unit_id == "FastLSTM" then
        rnn_unit = nn.FastLSTM(item_embeddings_size, item_embeddings_size)
    elseif rnn_unit_id == "GRU" then
        rnn_unit = nn.GRU(item_embeddings_size, item_embeddings_size)
    else
        error("Invalid RNN unit identifier!")
    end

    local items_model = nn.Sequential()
        :add(lookup_table)
        :add(nn.SplitTable(2))
        :add(nn.Sequencer(rnn_unit))
        -- average over rows of tensors in the table
        :add(nn.JoinTable(1))
        :add(nn.View(1, batch_size, item_embeddings_size))
        :add(nn.Mean(1))
        :add(nn.View(batch_size, item_embeddings_size))

    local users_model = nn.Sequential()
        :add(nn.LookupTable(num_users, user_embeddings_size))

    local full_model = nn.Sequential() -- {item, user}

    local parallel_table = nn.ParallelTable()
        :add(items_model)
        :add(users_model)

    if genres_data then
        print("-- Initializing model for genre features")
        hidden_dense_layer_size = hidden_dense_layer_size + genre_embeddings_size
        local genres_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#genres_data["genre2id"] + 1, genre_embeddings_size))
            :add(nn.Mean(2))

        parallel_table:add(genres_model)
    end

    full_model:add(parallel_table)
        :add(nn.JoinTable(2))
        :add(nn.Linear(hidden_dense_layer_size, 1))
        :add(cudnn.Sigmoid())

    return full_model
end

function build_model_amar_rnn_fast(items_data, ratings_data, genres_data, batch_size)
    local item_embeddings_size = 10
    local user_embeddings_size = 10
    local genre_embeddings_size = 10
    local hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
    local num_tokens = #items_data["token2id"]
    local num_users = #ratings_data["user2id"]
    local lookup_table = nn.LookupTableMaskZero(num_tokens, item_embeddings_size)

    local items_model = nn.Sequential()
        :add(lookup_table)
        :add(nn.SeqLSTM(item_embeddings_size, item_embeddings_size))
        :add(nn.Mean(2))

    local users_model = nn.Sequential()
        :add(nn.LookupTable(num_users, user_embeddings_size))

    local full_model = nn.Sequential() -- {item, user}

    local parallel_table = nn.ParallelTable()
        :add(items_model)
        :add(users_model)

    if genres_data then
        print("-- Initializing model for genre features")
        hidden_dense_layer_size = hidden_dense_layer_size + genre_embeddings_size
        local genres_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#genres_data["genre2id"] + 1, genre_embeddings_size))
            :add(nn.Mean(2))

        parallel_table:add(genres_model)
    end

    full_model:add(parallel_table)
        :add(nn.JoinTable(2))
        :add(nn.Linear(hidden_dense_layer_size, 1))
        :add(cudnn.Sigmoid())

    return full_model
end

function build_model_amar_cnn(items_data, ratings_data, genres_data, batch_size, max_item_len, num_filters, filter_size)
    local item_embeddings_size = 10
    local user_embeddings_size = 10
    local genre_embeddings_size = 10
    local hidden_dense_layer_size = num_filters + user_embeddings_size
    local num_tokens = #items_data["token2id"]
    local num_users = #ratings_data["user2id"]
    local lookup_table = nn.LookupTableMaskZero(num_tokens + 1, item_embeddings_size)

   local items_model = nn.Sequential()
      :add(lookup_table)
      :add(nn.Padding(1, -(num_filters-1), 2, 0))
      :add(nn.Padding(1, (num_filters-1), 2, 0))
      :add(cudnn.TemporalConvolution(item_embeddings_size, num_filters, filter_size))
      :add(cudnn.ReLU())
      :add(nn.TemporalMaxPooling(max_item_len+2*(num_filters-1)-filter_size+1))
      :add(nn.Squeeze(2))
      --:add(nn.Dropout(0.5)) ()

    local users_model = nn.Sequential()
        :add(nn.LookupTable(num_users, user_embeddings_size))

    local full_model = nn.Sequential() -- {item, user}

    local parallel_table = nn.ParallelTable()
        :add(items_model)
        :add(users_model)

    if genres_data then
        print("-- Initializing model for genre features")
        hidden_dense_layer_size = hidden_dense_layer_size + genre_embeddings_size
        local genres_model = nn.Sequential()
            :add(nn.LookupTableMaskZero(#genres_data["genre2id"] + 1, genre_embeddings_size))
            :add(nn.Mean(2))

        parallel_table:add(genres_model)
    end

    full_model:add(parallel_table)
        :add(nn.JoinTable(2))
        :add(nn.Linear(hidden_dense_layer_size, 1))
        :add(cudnn.Sigmoid())

    return full_model
end

