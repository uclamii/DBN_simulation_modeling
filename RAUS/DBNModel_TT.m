% Make and train a DBN from data
%
% Input:
% dataTrainComplete : 2d array of training data no missing values
% dataTrain: Training data
% dataTrainMiss: training data and missing data
% dataValid: Validation/test data
% ns: array that depicts number of categories per variable
% dag: intra-structure, 2d array (number of variables, number of variables)
% max_fan_in: integer number of allowed edges between variables
% intraLength: integer number of variables
% horizon: integer number of time points
%
% Output:
% 2d - array of probabilities for training and testing data.


function dataTrainValid = DBNModel_TT(inter_structure, dataTrain, dataTrainMiss, dataValid, ns, dag, max_iter, intraLength, horizon, numNodes,ncases) %add dataTest to this function

          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          clc;
          warning('off', 'Octave:possible-matlab-short-circuit-operator');
      disp('Setting Path');
          %%%%%%%%%%%%%%%%%%%% get path to BNT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          origPath = pwd;
          cd ./BNT
          addpath(genpathKPM(pwd))
          cd(origPath)
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Loading Learned Structures');
          %%%%%%%%%%%%%%%%%% learn interslice structure of DBN %%%%%%%%%%%%%%%%%%%%%%%%%%%
          inter2 = inter_structure;
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%%%% DBN - K2 Learned %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          intra2 = dag;
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Creating DBN');
          %%%%%%%%%%%%%%%%%%%%% DBN creation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          bnet = createDBN(intraLength, intra=intra2, inter=inter2, ns, numNodes);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %disp('Visualizing DBN');
          %draw_dbn(intra2,inter2);

          %%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          ss = intraLength;%slice size(ss)
          T = horizon;


          casesTrain = data2cell(dataTrain, ss, T,to_replace=-1);
	      casesTrainMiss = data2cell(dataTrainMiss, ss, T, to_replace=-1);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


          %%%%%%%%%%%%%%%%%%% engine definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          engine = smoother_engine(jtree_2TBN_inf_engine(bnet));


          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Learning DBN Parameters, Which May Take Some Time...')
          %%%%%%%%%%%%%%%%%%% learn dbn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          [bnet2, LLtrace] = learn_params_dbn_em(engine, casesTrain,'max_iter', max_iter);

          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Starting Inference, Which May Take Some Time...');
          %%%%%%%%%%%%%%%%%%%%%% Set evidence and learn marginals %%%%%%%%%%%%%%%%%%%%%%%%
          dataTrainMarginals = dbnInference(casesTrainMiss, ss, T, ncases=size(dataTrainMiss)(1), bnet=bnet2, numNodes);
          %dataTrainMarginals = dbnInference(casesTrainMiss, ss, T, ncases, bnet=bnet2, numNodes); %uncomment to run inference on subset of ncases


          casesValid = data2cell(dataValid, ss, T,to_replace=-1);
          dataValidMarginals = dbnInference(casesValid, ss, T, ncases=size(dataValid)(1), bnet=bnet2, numNodes);
          %dataValidMarginals = dbnInference(casesValid, ss, T, ncases, bnet=bnet2, numNodes); %uncomment to run inference on subset of ncases


          dataTrainValid = [dataTrainMarginals; dataValidMarginals];

end
