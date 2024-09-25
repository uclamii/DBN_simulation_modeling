% Make a BN and return marginal probabilities

function dataTrainValid = BNModelLearn_TVT(dag, intraLength, numNodes, max_iter, dataTrain, dataValid, dataTest,ns,horizon,ncases,onlyValid)

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
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Creating BN');
          %%%%%%%%%%%%%%%%%%%%%% BN creation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          bnet = createBN(intraLength, dag, ns, numNodes);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%% engine definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      disp('Engine Set');
          engine = jtree_inf_engine(bnet);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          ss = intraLength;%slice size(ss)
          T = horizon;

          casesTrain = bndata2cell(dataTrain, ns,ss, to_replace=-1);
          %casesTrain = cell(dataTrain,ss)

          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Learning BN Parameters, Which May Take Some Time...');
          %%%%%%%%%%%%%%%%%%%% learn bn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          [bnet2, LLtrace] = learn_params_em(engine,casesTrain,'max_iter',max_iter);
          %[bnet2, LLtrace] = learn_params_em(engine, casesTrain,'max_iter',max_iter);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      disp('Starting Inference, Which May Take Some Time...');
      if onlyValid==1
          casesValid = bndata2cell(dataValid, ns,ss, to_replace=-1);
          %casesValid = cell(dataValid,ss)
          dataValidMarginals = bnInference(casesValid, ns,ss, numNodes, ncases=size(dataValid)(1), bnet=bnet2);
          %dataValidMarginals = bnInference(casesValid, ns,ss, numNodes, ncases, bnet=bnet2);
          dataTrainValid = dataValidMarginals;
      else
          %%%%%%%%%%%%%%%%%%%%%% Set evidence and learn marginals %%%%%%%%%%%%%%%%%%%%%%%%
          dataTrainMarginals = bnInference(casesTrain, ns,ss, numNodes, ncases=size(dataTrain)(1), bnet=bnet2);
          %dataTrainMarginals = bnInference(casesTrain, ns,ss, numNodes, ncases, bnet=bnet2);


          casesValid = bndata2cell(dataValid, ns,ss, to_replace=-1);
          %casesValid = cell(dataValid,ss)
          dataValidMarginals = bnInference(casesValid, ns,ss, numNodes, ncases=size(dataValid)(1), bnet=bnet2);
          %dataValidMarginals = bnInference(casesValid, ns,ss, numNodes, ncases, bnet=bnet2);

          casesTest = bndata2cell(dataTest, ns,ss, to_replace=-1);
          %casesTest = cell(dataTest,ss)
          dataTestMarginals = bnInference(casesTest, ns,ss, numNodes, ncases=size(dataTest)(1), bnet=bnet2);
          %dataTestMarginals = bnInference(casesTest, ns,ss, numNodes, ncases, bnet=bnet2);

          dataTrainValid = [dataTrainMarginals; dataValidMarginals; dataTestMarginals];
      endif
end
