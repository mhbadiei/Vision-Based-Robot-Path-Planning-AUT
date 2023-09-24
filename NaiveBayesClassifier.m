function predictedLable = NaiveBayesClassifier(trainData, trainLable, testData, testLable, distr)
yu=unique(trainLable);
nc=length(yu);
ni=size(trainData,2);
ns=length(testLable);
for i=1:nc
    fy(i)=sum(double(trainLable==yu(i)))/length(trainLable);
end
switch distr
    case 'normal'
        for i=1:nc
            xi=trainData((trainLable==yu(i)),:);
            mu(i,:)=mean(xi,1);
            sigma(i,:)=std(xi,1);
        end
        for j=1:ns
            fu=normcdf(ones(nc,1)*testData(j,:),mu,sigma);
            P(j,:)=fy.*prod(fu,2)';
        end
    case 'kernel'
        for i=1:nc
            for k=1:ni
                xi=trainData(trainLable==yu(i),k);
                ui=testData(:,k);
                fuStruct(i,k).f=ksdensity(xi,ui);
            end
        end
        for i=1:ns
            for j=1:nc
                for k=1:ni
                    fu(j,k)=fuStruct(j,k).f(i);
                end
            end
            P(i,:)=fy.*prod(fu,2)';
        end
    otherwise
        disp('invalid distribution stated')
        return
end
[pv0,id]=max(P,[],2);
for i=1:length(id)
    predictedLable(i,1)=yu(id(i));
end
end