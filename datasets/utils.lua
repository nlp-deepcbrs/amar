local tds = require "tds";

--[[
Adds padding values to the input sequences according to the max_sequence_len parameter.

Parameters:
    - sequences: list of tensors to be padded
    - max_sequence_len: maximum number of element that each sequence should have
 ]]
function pad_sequences(sequences, max_sequence_len)
    local data = torch.Tensor(#sequences, max_sequence_len):zero()

    for i = 1, #sequences do
        if sequences[i]:dim() == 1 then
            for j = 1, sequences[i]:size(1) do
                data[i][j] = sequences[i][j]
            end
        end
    end

    return data
end

--[[
  Loads the dense continuous representation from a specific embeddings filename which
  represents embeddings in a specific type, such as GloVe or Word2Vec.

  Parameters:
    - embeddings_filename: filename of the embeddings
    - embedding_type: embedding type identifier: {"word2vec", "glove"}
 ]]
function load_embeddings(embeddings_filename, embeddings_type)
    function load_word2vec_embeddings(embeddings_filename)
        file = torch.DiskFile(embeddings_filename, "r")
        local max_w = 50

        function readStringv2(file)
            local str = {}

            for i = 1, max_w do
                local char = file:readChar()

                if char == 32 or char == 10 or char == 0 then
                    break
                else
                    str[#str + 1] = char
                end
            end

            str = torch.CharStorage(str)
            return str:string()
        end

        -- reading header
        file:ascii()
        num_words = file:readInt()
        embedding_size = file:readInt()
        local embeddings = tds.Hash()

        -- reading content
        file:binary()
        for i = 1, num_words do
            local word = readStringv2(file)
            local word_embedding = file:readFloat(embedding_size)
            word_embedding = torch.FloatTensor(word_embedding)

            local norm = torch.norm(word_embedding, 2)

            -- normalize word embedding
            if norm ~= 0 then
                word_embedding:div(norm)
            end

            embeddings[word] = word_embedding
        end

        return {
            embeddings = embeddings,
            embedding_size = embedding_size
        }
    end

    function load_glove_embeddings(embeddings_filename)
        local embeddings = tds.Hash()
        local delimiter = " "
        local data = file.read(embeddings_filename)
        local file_lines = stringx.splitlines(data)
        local embedding_size = #stringx.split(file_lines[1], delimiter) - 1

        for i = 1, #file_lines do
            local splitted_line = stringx.split(file_lines[i], delimiter)
            local word = splitted_line[1]
            local word_embedding = torch.Tensor(embedding_size)

            for j = 2, #splitted_line do
                word_embedding[j - 1] = tonumber(splitted_line[j])
            end

            local norm = torch.norm(word_embedding, 2)

            -- normalize word embedding
            if norm ~= 0 then
                word_embedding:div(norm)
            end

            embeddings[word] = word_embedding
        end

        return {
            embeddings = embeddings,
            embedding_size = embedding_size
        }
    end

    if embeddings_type == "word2vec" then
        return load_word2vec_embeddings(embeddings_filename)
    elseif embeddings_type == "glove" then
        return load_glove_embeddings(embeddings_filename)
    else
        error("Invalid embedding type!")
    end
end
