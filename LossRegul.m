classdef LossRegul < dagnn.Loss
%LOSSREGUL Loss function for regularisation (L1) of features X
%   Detailed explanation goes here

properties
    type = 'L1'
end

methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnlossregul(inputs{1}, inputs{2}, [], obj.type) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnlossregul(inputs{1}, inputs{2}, derOutputs{1}, obj.type) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = LossRegul(varargin)
      obj.load(varargin) ;
    end
end
end
