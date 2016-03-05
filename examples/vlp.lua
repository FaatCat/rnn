--[[
Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.
]]--

require 'rnn'

version = 1.2 -- refactored numerical gradient test into unit tests. Added training loop

local opt = {}
opt.data_dir = "/home/ubuntu/fyp/python/top_output_all.txt" --"/home/ubuntu/fyp/python/top_class_output.txt"
opt.learningRate = 0.1
opt.hiddenSize = 256
opt.vocabSize = 234 -- classes (ascii max)
opt.seqLen = 195 -- length of the encoded sequence
opt.niter = 35000
opt.batchsize = 64
opt.training_split = 0.7
opt.output_len = 10 -- unused
opt.test_number = 3 -- number of items to test each time
opt.gpu = true

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpu then
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
   cutorch.setDevice(1) -- use first gpu found
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
local function forwardConnect(encLSTM, decLSTM)
   decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.seqLen])
   decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.seqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
local function backwardConnect(encLSTM, decLSTM)
   encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
   encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

-- Encoder
local enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
enc:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local encLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
enc:add(nn.Sequencer(encLSTM))
enc:add(nn.SelectTable(-1))
if opt.gpu then enc:cuda() end

-- Decoder
local dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local decLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSize, opt.vocabSize)))
dec:add(nn.Sequencer(nn.LogSoftMax()))
if opt.gpu then dec:cuda() end

local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
if opt.gpu then criterion:cuda() end

local ord = string.byte
local chr = string.char

-- Some example data (batchsize = 2)
--local encInSeq = torch.CudaTensor({{1,2,3,5,4,1,2,4,10},{3,2,1,4,5,13,24,23,31}}) 
local function read_data_top_output(filename)
   local outputs = {}
   local labels = {}
   local file = io.open(filename, 'r')
   for line in file:lines() do
      local start_, end_ = string.find(line,';')
      local raw_label = string.sub(line, 1, end_-1)
      local raw_output = string.sub(line, end_+1)
      local label = {}
      local output = {}
      for c in raw_label:gmatch"." do
         table.insert(label, ord(c))
      end
      for c in raw_output:gmatch('([^,]+)') do
         table.insert(output, ord(c))
      end
      table.insert(outputs, output)
      table.insert(labels, label)
     print('Read: ' .. raw_label)
   end
   return outputs, labels
end


local function forward(model, inSeq)
   local encInSeq = inSeq
   local enc = model[1]
   local dec = model[2]
   local encLSTM = model[3]
   local decLSTM = model[4]
   --local decInSeq = torch.CudaTensor({{1,1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1,1}})
   
  
   enc:zeroGradParameters()
   dec:zeroGradParameters()
   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM) -- Copy encoder output into decoder input
   local decOut = dec:forward(decInSeq)

   local batchSize = 1
   if (1==1) then
     --print(inputs)
     local batch_i = 1
       for batch_i=1, batchSize do
       local outseq = ''
       for i=1,#decOut do
         val, index = torch.max(decOut[i][batch_i], 1)
         outseq = outseq .. ',' .. chr(index[1])
       end
       local inseq = ''
       --print(encInSeq:size(2))
       for i=1, encInSeq:size(2) do
         inseq = inseq .. ',' ..  chr(encInSeq[batch_i][i])
       end
       --print(batch_i .. ': [IN]' .. inseq .. ' -> [OUT]' .. outseq)
       local prettyOut = outseq
       local prettyIn = inseq
       return prettyIn, prettyOut
     end
   end
end

local function save_model(name, enc, dec, encLSTM, decLSTM)
   local model = {enc, dec, encLSTM, decLSTM}
   torch.save(name, model)
end

print('Loading data...')
local data, labels = read_data_top_output(opt.data_dir)
local training_data_split_index = math.floor(#data * opt.training_split) --all items before this index are used in training data
print(#data .. ' data loaded. First ' .. training_data_split_index .. ' used for training.')

--local encInSeq = torch.CudaTensor({{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2')},{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7')}})
local decoderInput = {}
for i=1, opt.batchsize do
   table.insert(decoderInput, {1,1,1,1,1,1,1,1,1,1})
end

local decInSeq = {}
if opt.gpu then 
   decInSeq = torch.CudaTensor(decoderInput)
else
   decInSeq = torch.Tensor(decoderInput)
end

local start_time = os.time()
for iteration=1,opt.niter do
   local batch_data = {}
   local label = {}
   local validating = (iteration%100==0)
   for i=1, opt.batchsize do 
      local random_index = math.random(training_data_split_index)
	  if (validating) then random_index = training_data_split_index + math.random(#data - training_data_split_index) end
      table.insert(batch_data, data[random_index])
      table.insert(label, labels[random_index])
   end
   local batch = {}
   if opt.gpu then 
      batch = torch.CudaTensor(batch_data)
   else
      batch = torch.Tensor(batch_data)
   end
--local decOutSeq = torch.CudaTensor({{ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('F'),ord('2'),ord('2'),ord('2')},{ord('0'),ord('0'),ord('0'),ord('W'),ord('T'),ord('G'),ord('5'),ord('7'),ord('6'), ord('7')}})
   local target_sequence = {}
   if opt.gpu then
      target_sequence = torch.CudaTensor(label)
   else 
      target_sequence = torch.Tensor(label)
   end
   
   if opt.gpu then 
      target_sequence = nn.SplitTable(1, 1):cuda():forward(target_sequence) 
   else 
      target_sequence = nn.SplitTable(1, 1):forward(target_sequence)
   end

   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Forward pass
   local encOut = enc:forward(batch)
   forwardConnect(encLSTM, decLSTM) -- Copy encoder output into decoder input
   local decOut = dec:forward(decInSeq)
   local err = criterion:forward(decOut, target_sequence)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   
   local test_number = opt.test_number
   
   if (iteration%50==0) then
      --print(inputs)
      for batch_i=1, test_number do
         local inseq = ''
         for i=1, batch:size(2) do
            inseq = inseq .. '' ..  chr(batch[batch_i][i])
         end
         local outseq = ''
         for i=1,#decOut do
            val, index = torch.max(decOut[i][batch_i], 1)
            outseq = outseq .. '' .. chr(index[1])
         end
         local targetseq = ''
         for i=1,#target_sequence do
            targetseq = targetseq .. '' .. chr(target_sequence[i][batch_i])
         end
		 if (validating) then print('VALIDATION SET') end
         print(batch_i .. ': [IN]' .. inseq .. ' -> [OUT]' .. outseq .. ' [TRUTH]'.. targetseq ..'')
		 print('Average time per iteration: ' .. (os.difftime(os.time(), start_time)/iteration))
      --print(targets)
      end
   end
   if (iteration%1000 == 0) then
      save_model("model_iter" .. iteration .. '.t7', enc, dec, encLSTM, decLSTM)
     print('Saving model model_iter' .. iteration .. '.t7')
     if opt.learningRate > 0.001 then opt.learningRate = opt.learningRate / 10 end
     print ('Learning rate: ' .. opt.learningRate)
   end

   if not validating then
      -- Backward pass
      local gradOutput = criterion:backward(decOut, target_sequence)
      dec:backward(decInSeq, gradOutput)
      backwardConnect(encLSTM, decLSTM)
      local zeroTensor = {}
	  if opt.gpu then zeroTensor = torch.CudaTensor(2):zero() else zeroTensor = torch.Tensor(2):zero() end
      enc:backward(batch, zeroTensor)
      dec:updateParameters(opt.learningRate)
      enc:updateParameters(opt.learningRate)
   else
      iteration = iteration - 1
   end
end

save_model("endmodel.t7", enc, dec, encLSTM, decLSTM)


model = torch.load("endmodel.t7")
print('Evaluating...')

forward(model, torch.CudaTensor{{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('P'),ord('P'),ord('N'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3')},{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('P'),ord('P'),ord('N'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3')}})
forward(model, data[training_data_split_index + math.random(#data - training_data_split_index)])

