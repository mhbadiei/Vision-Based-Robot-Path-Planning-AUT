function predictedLable = KNN_Classifier(k,trainData,trainLable,testData)
if size(trainData,2)~=size(testData,2)
    error('data should have the same dimensionality');
end
if mod(k,2)==0
    error('to reduce the chance of ties, please choose odd k');
end
predictedLable=zeros(size(testData,1),1);
ed=zeros(size(testData,1),size(trainData,1)); 
ind=zeros(size(testData,1),size(trainData,1));
k_nn=zeros(size(testData,1),k); 
for test_point=1:size(testData,1)
    for train_point=1:size(trainData,1)
        ed(test_point,train_point)=sqrt(...
            sum((testData(test_point,:)-trainData(train_point,:)).^2));
    end
    [ed(test_point,:),ind(test_point,:)]=sort(ed(test_point,:));
end
k_nn=ind(:,1:k);
nn_index=k_nn(:,1);
for i=1:size(k_nn,1)
    options=unique(trainLable(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(trainLable(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predictedLable(i)=max_label;
end