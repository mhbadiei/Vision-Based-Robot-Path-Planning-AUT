function predictedLable = SVM_Classifier(trainData,trainLable,testData)
    cl = fitcsvm(trainData,trainLable,'KernelFunction','rbf',...
        'BoxConstraint',Inf,'ClassNames',[-1,1]);
    %[predictedLable,scores] = predict(cl,testData);
    predictedLable = predict(cl,testData);
end