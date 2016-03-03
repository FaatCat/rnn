--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'rnn'

version = 1.2 -- refactored numerical gradient test into unit tests. Added training loop

local opt = {}
opt.learningRate = 0.1
opt.hiddenSize = 32
opt.vocabSize = 234
opt.seqLen = 196 -- length of the encoded sequence
opt.niter = 500

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

local ord = string.byte
local chr = string.char

-- Some example data (batchsize = 2)
--local encInSeq = torch.Tensor({{1,2,3,5,4,1,2,4,10},{3,2,1,4,5,13,24,23,31}}) 
local encInSeq = torch.Tensor({{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('F'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2')},{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('W'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('G'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('5'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('6'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('7'),ord('7'),ord('0'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7'),ord('7')}})
local decInSeq = torch.Tensor({{1,1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1,1}})
local decOutSeq = torch.Tensor({{ord('0'),ord('0'),ord('0'),ord('0'),ord('W'),ord('W'),ord('F'),ord('2'),ord('2'),ord('2')},{ord('0'),ord('0'),ord('0'),ord('W'),ord('T'),ord('G'),ord('5'),ord('7'),ord('6'), ord('7')}})
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
   if (iteration%10==0) then
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
         local targetseq = ''
         for i=1,#decOutSeq do
            targetseq = targetseq .. ',' .. chr(decOutSeq[i][batch_i])
         end
         print(batch_i .. ': [IN]' .. inseq .. ' -> [OUT]' .. outseq .. '([TRUTH]'.. targetseq ..')')
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

local model = {enc, dec, encLSTM, decLSTM}
torch.save("model.t7", model)
model = torch.load("model.t7")

local function eval(model, inSeq)
        local encInSeq = inSeq
        local enc = model[1]
        local dec = model[2]
        local encLSTM = model[3]
        local decLSTM = model[4]
        local decInSeq = torch.Tensor({{1,1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1,1}})
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
		 print(batch_i .. ': [IN]' .. inseq .. ' -> [OUT]' .. outseq)
	  end
	end
end

print('Evaluating...')

eval(model, torch.Tensor{{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('P'),ord('P'),ord('N'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3')},{ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('B'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('P'),ord('P'),ord('N'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('M'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('C'),ord('T'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('2'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('0'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3'),ord('3')}})
