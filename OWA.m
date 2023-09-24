close; clear;clc

disp('--- start ---')

itr = 100;

distr='kernel';
accuracy=zeros(1,itr);
precision=zeros(1,itr);

itr = 100;

distr='kernel';

accuracy=zeros(1,itr);
average_accuracy_of_accuracy = zeros(5,1);
standard_deviation_of_accuracy = zeros(5,1);

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
    [predictedLable_1] = NaiveBayesClassifier(trainData, trainLable, testData, testLable, distr);
    
    [predictedLable_2] = KNN_Classifier(3,trainData, trainLable, testData);
    
    [predictedLable_3] = KNN_Classifier(5,trainData, trainLable, testData);
    
    [predictedLable_4] = SVM_Classifier(trainData, 2.*(trainLable-1.5), testData);
    predictedLable_4 = 1.5 + predictedLable_4./2;
    
    [predictedLable_5] = decisionTreeClassifier(trainData, trainLable, testData);
    
    predicted = [2*(predictedLable_1 -1.5), 2*(predictedLable_2-1.5), 2*(predictedLable_3-1.5), 2*(predictedLable_4-1.5), 2*(predictedLable_5-1.5)];
    [B, OrigColIdx] = sort(predicted,2,'descend');
    predictedLable = sign(5307*B(:,1) + 0.2565*B(:,2) + 1240*B(:,3) + 0.0599*B(:,4) + 0.0290*B(:,5));
    accuracy(n) = sum(predictedLable==2*(testLable-1.5))/length(predictedLable);
end
str = {'OHagan'};

figure()
ax = axes;
plot(1:1:itr,100.*accuracy,'LineWidth',3)
title('Accuracy Curve Of OWA Based on Iterations')
xlabel('ITERATION','fontweight','bold','fontsize',10);
ylabel('ACCURACY (%)','fontweight','bold','fontsize',10);
set(ax, 'YTick', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'YLim', [0, 100]);
ytickformat(ax, 'percentage');
legend(str)

average_accuracy_of_accuracy = mean(accuracy);
standard_deviation_of_accuracy = std(accuracy);

