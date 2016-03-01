--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'rnn'

version = 1.2 -- refactored numerical gradient test into unit tests. Added training loop

local opt = {}
opt.learningRate = 0.1
opt.hiddenSize = 6
opt.vocabSize = 34
opt.seqLen = 9 -- length of the encoded sequence
opt.niter = 150

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

-- Decoder
local dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local decLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSize, opt.vocabSize)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Some example data (batchsize = 2)
local encInSeq = torch.Tensor({{1,2,3,5,4,1,2,4,10},{3,2,1,4,5,13,24,23,31}}) 
local decInSeq = torch.Tensor({{1,1,1,1,1},{1,1,1,1,1}})
local decOutSeq = torch.Tensor({{2,3,4,1,5},{1,2,4,3,5}})
decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

for iteration=1,opt.niter do
   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM) -- Copy encoder output into decoder input
   local decOut = dec:forward(decInSeq)
   local err = criterion:forward(decOut, decOutSeq)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   local batchSize = 2
   if (iteration%1==0) then
      --print(inputs)
      local batch_i = 1
         for batch_i=1, batchSize do
         local outseq = ''
         for i=1,#decOut do
            val, index = torch.max(decOut[i][batch_i], 1)
            outseq = outseq .. ', ' .. index[1]
         end
         local inseq = ''
         --print(encInSeq:size(2))
         for i=1, encInSeq:size(2) do
            inseq = inseq .. ',' ..  encInSeq[batch_i][i]
         end
         local targetseq = ''
         for i=1,#decOutSeq do
            targetseq = targetseq .. ',' .. decOutSeq[i][batch_i]
         end
         print(batch_i .. ': ' .. inseq .. ' -> ' .. outseq .. '('.. targetseq ..')')
      --print(targets)
      end
   end


   -- Backward pass
   local gradOutput = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gradOutput)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.Tensor(2):zero()
   enc:backward(encInSeq, zeroTensor)

   dec:updateParameters(opt.learningRate)
   enc:updateParameters(opt.learningRate)
end
