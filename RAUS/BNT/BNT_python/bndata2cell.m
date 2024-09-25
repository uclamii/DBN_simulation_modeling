function cases = bndata2cell(data, ns,ss, to_replace)
          % function that takes an input matrix and transforms into cell2mat
          %input to the BNT dbn learn, inference and structure learn library
          % see https://www.cs.ubc.ca/~murphyk/Software/BNT/usage_02nov13.html under mixture of experts header
          % see https://www.cs.utah.edu/~tch/notes/matlab/bnt/docs/usage.html#basics under mixture of experts for transposing the input data matrix (so its (ncolumns,ncases)) for input to the learn_params_em function
          data = data';%matrix needs to be transposed for learn_params_em function, specifically for line 66-68.
          data(isnan(data)) = -1;

          sizeData = size(data);
          ncases = sizeData(2);
          ncolumns = sizeData(1);


          cases = cell(ncolumns,ncases);
          %disp(size(data));
          %disp(size(cases));


          %cases = num2cell(data');

          %cases = cell(1, ncases);
          %for j=1:ncases
            %cases{j} = cell(ss);
          for i=1:ncolumns
            for j=1:ncases
              if data(i,j)==to_replace
                cases{i,j} = [];
              else
                cases(i,j) = data(i,j);
              %disp(data);
              %disp(cases);
                %disp(size(cases));
                %disp(size(data));
              end
            end
          end
end
