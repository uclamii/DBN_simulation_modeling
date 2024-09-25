function dataMarginals = bnInference(cases, ns,ss,numNodes,ncases, bnet)
          %function that takes as input cases as cells and computes the marginal
          %probability of a node given evidence at each time point P(x_t|y0,y1,.._y_t)
          %returns a matrix of all marginal probabilites of of a binary hidden node

          % cases are the data, cases are cell structures that contain cell structures
          %for example each cell is a data point and each cell inside that cell
          %is the data for each node of the network over time
          % s is the slize size
          % ncases is the number of cases
          % bnet is the parameterized network

         %onodes = [1,3:numNodes];
         node_i = 2; %outcome node

         dataMarginals = zeros(ncases,1);

         for i=1:ncases
          %   for t=1:T
             %evidence = cases{i}(1:ss);
             evidence = cases(:,i);

	     %for j=1:t
	         evidence{node_i} = [];
	     %end

         %evidence = cell(1,ss);

	     engine = jtree_inf_engine(bnet);

	     [engine, ll] = enter_evidence(engine, evidence);

	     marg = marginal_nodes(engine, node_i);
	     dataMarginals(i) = marg.T(2);
             %end
         end

end
