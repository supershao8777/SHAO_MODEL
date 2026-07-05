function [resmax]= myNMIACCwithmean_xxl(U,Y,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U;
maxIter = 50;

for iter = 1:maxIter
    indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
%     indx = kmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
    indx = indx(:);
    result(iter,:) = Clustering8Measure(Y,indx);
end
resmax = max(result,[],1);