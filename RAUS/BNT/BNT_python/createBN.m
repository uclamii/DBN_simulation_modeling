function bnet = createBN(intraLength, dag, ns, numNodes)
          %function that creates a bn and then instantiates random CPT tables
         dnodes = 1:intraLength;
         bnet = mk_bnet(dag, ns, 'discrete', dnodes);
         %bnet = mk_bnet(dag, ns);

         % nodes initialization - randomized
         for i=1:numNodes
           bnet.CPD{i} = tabular_CPD(bnet, i,'CPT','rnd');
           %bnet.CPD{i} = tabular_CPD(bnet, i);
         end
end