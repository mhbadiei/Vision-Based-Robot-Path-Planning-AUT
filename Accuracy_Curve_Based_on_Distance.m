close; clear;clc

disp('--- start ---')

itr = 85;

distr='kernel';
accuracy= zeros(5,itr);
precision= zeros(5,itr);
distance= zeros(1,itr);
for i = 0:1:itr
    i
    n = i+1;
    randn('seed',n)
    m1 = rand(3,1)';
    mu = [m1 ;m1+0.02*i.*ones(3,1)]';
    %sqrt(abs(sum((mu(:,1)-mu(:,2)).^2)))
    distance(1,n) = sqrt(abs(sum((mu(:,1)-mu(:,2)).^2)));
    Sigma_x = 0.2*eye(3);
    Sigma_y = 0.4*eye(3);
    points_per_class = [500 500];
    X1 = mvnrnd(mu(:,1),Sigma_x,points_per_class(1))';
    X1 = [X1 mvnrnd(mu(:,2), Sigma_y, points_per_class(2))'];
    y1 = [ones(1, 350) -ones(1,350)];
    trainData = [X1(:,1:350)';X1(:,501:850)'];
    trainLable = y1';
    testData = [X1(:,351:500)';X1(:,851:1000)'];
    testLable = [ones(1, 150) -ones(1,150)]';
    [predictedLable] = NaiveBayesClassifier(trainData, trainLable, testData, testLable, distr);
    accuracy(1,n) = sum(predictedLable==testLable)/length(predictedLable);
    precision(1,n) = sum(predictedLable(1:150)==1)/sum(predictedLable==1);
    
    [predictedLable] = KNN_Classifier(3,trainData, trainLable, testData);
    accuracy(2,n) = sum(predictedLable==testLable)/length(predictedLable);
    precision(2,n) = sum(predictedLable(1:150)==1)/sum(predictedLable==1);
    
    [predictedLable] = KNN_Classifier(5,trainData, trainLable, testData);
    accuracy(3,n) = sum(predictedLable==testLable)/length(predictedLable);
    precision(3,n) = sum(predictedLable(1:150)==1)/sum(predictedLable==1);
    
    [predictedLable] = SVM_Classifier(trainData, trainLable, testData);
    accuracy(4,n) = sum(predictedLable==testLable)/length(predictedLable);
    precision(4,n) = sum(predictedLable(1:150)==1)/sum(predictedLable==1);
    
    [predictedLable] = decisionTreeClassifier(trainData, trainLable, testData);
    accuracy(5,n) = sum(predictedLable==testLable)/length(predictedLable);
    precision(5,n) = sum(predictedLable(1:150)==1)/sum(predictedLable==1);
end
str = {'Naive Bayes', 'KNN K=3', 'KNN K=5', 'SVM','Decision Tree'};
for i=1:1:5
    figure()
    ax = axes;
    plot(distance,100.*accuracy(i,:),'LineWidth',3)
    title('Accuracy Based on Distance')
    xlabel('DISTANCE','fontweight','bold','fontsize',10);
    ylabel('ACCURACY (%)','fontweight','bold','fontsize',10);
    set(ax, 'YTick', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'YLim', [0, 100]);
    ytickformat(ax, 'percentage');
    legend(str{i})
    figure()
    ax = axes;
    plot(distance,100.*precision(i,:),'LineWidth',3)
    title('Precision Based on Distance');
    xlabel('DISTANCE','fontweight','bold','fontsize',10);
    ylabel('PRECISION (%)','fontweight','bold','fontsize',10);
    set(ax, 'YTick', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'YLim', [0, 100]);
    ytickformat(ax, 'percentage');
    legend(str{i})
end
