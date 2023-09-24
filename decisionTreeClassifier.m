function predictedLable = decisionTreeClassifier(trainData,trainLable,testData)
model = fitctree(trainData,trainLable,'MinParentSize',2); 
predictedLable = predict(model,testData);
end