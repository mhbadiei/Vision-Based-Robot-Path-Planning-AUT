close; clear;clc

disp('--- start ---')

itr = 100;

distr='kernel';
accuracy=zeros(5,itr);
precision=zeros(5,itr);
average_accuracy_of_accuracy = zeros(5,1);
standard_deviation_of_accuracy = zeros(5,1);
average_accuracy_of_precision = zeros(5,1);
standard_deviation_of_precision = zeros(5,1);

%% retrieving data from Data.xlsx and storing in A matrix.
[data, text] = xlsread('data\Data.xlsx');
A = data;
X = A(:,(1:3));
y = A(:,4);

for n = 1:1:itr
    data = A;
    cv = cvpartition(size(data,1),'HoldOut',0.3);
    idx = cv.test;
    dataTrain = data(~idx,:)';
    dataTest  = data(idx,:)';
    
    dataTrain = dataTrain';
    dataTest = dataTest';
    
    trainData = dataTrain(:,(1:3));
    trainLable = dataTrain(:,4);
    testData = dataTest(:,(1:3));
    testLable = dataTest(:,4);
    [predictedLable] = NaiveBayesClassifier(trainData, trainLable, testData, testLable, distr);
    accuracy(1,n) = sum(predictedLable==testLable)/length(predictedLable);
    
    [predictedLable] = KNN_Classifier(3,trainData, trainLable, testData);
    accuracy(2,n) = sum(predictedLable==testLable)/length(predictedLable);
    
    [predictedLable] = KNN_Classifier(5,trainData, trainLable, testData);
    accuracy(3,n) = sum(predictedLable==testLable)/length(predictedLable);
    
    [predictedLable] = SVM_Classifier(trainData, 2.*(trainLable-1.5), testData);
    accuracy(4,n) = sum(1.5+predictedLable/2==testLable)/length(predictedLable);
    
    [predictedLable] = decisionTreeClassifier(trainData, trainLable, testData);
    accuracy(5,n) = sum(predictedLable==testLable)/length(predictedLable);
end
str = {'Naive Bayes', 'KNN K=3', 'KNN K=5', 'SVM','Decision Tree'};
for i=1:1:5
    figure()
    ax = axes;
    plot(1:1:itr,100.*accuracy(i,:),'LineWidth',3)
    title('Accuracy Curve Based on Iterations')
    xlabel('ITERATION','fontweight','bold','fontsize',10);
    ylabel('ACCURACY (%)','fontweight','bold','fontsize',10);
    set(ax, 'YTick', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'YLim', [0, 100]);
    ytickformat(ax, 'percentage');
    legend(str{i})
end
