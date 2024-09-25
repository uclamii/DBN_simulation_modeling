% Learn Inter-structure
% 
% Input:
% dataTrainComplete : 2d array of training data no missing values
% ns: array that depicts number of categories per variable
% max_fan_in: integer number of allowed edges between variables
% intraLength: integer number of variables
% horizon: integer number of time points
%
% 1) Learn inter-strucure, using all time points of horizon, algorithm (EM)
% requires complete data, filter out cases with missing data.
% 
%
% Output:
% inter-structure


function inter2 = interStructLearn(dataTrainComplete, ns, max_fan_in, intraLength, horizon)

          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          clc;
          warning('off', 'Octave:possible-matlab-short-circuit-operator');
          %%%%%%%%%%%%%%%%%%%% get path to BNT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          origPath = pwd;
          cd ./BNT
          addpath(genpathKPM(pwd))
          cd(origPath)
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          ss = intraLength;%slice size(ss)
          T = horizon;

          casesTrainComplete = data2cell(dataTrainComplete, ss, T,to_replace=-1);

      %disp(size(casesTrainComplete));
      disp('Learning Inter-Structure');
          %%%%%%%%%%%%%%%%%% learn interslice structure of DBN %%%%%%%%%%%%%%%%%%%%%%%%%%%
          inter2 = learn_struct_dbn_reveal(casesTrainComplete, ns, max_fan_in);
          
end
