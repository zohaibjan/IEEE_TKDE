function Ensemble_decision=CombineCLFs(CombinitionMethod,...
  CLF_Train_output,CLF_Test_output,N_classifiers,N_test,N_train,N_class,TrainTargets)
%% Make necessary variable
% Train_abstract_output=zeros(N_classifiers,N_train);
% Train_rank_output=zeros(N_classifiers,N_class,N_train);
Train_DP_output=zeros(N_classifiers,N_class,N_train);
Test_abstract_output=zeros(N_classifiers,N_test);
Test_rank_output=zeros(N_classifiers,N_class,N_test);
Test_DP_output=zeros(N_classifiers,N_class,N_test);
Confusion_Matrx=zeros(N_class,N_class,N_classifiers);
for c=1:N_classifiers
  %Train_abstract_output(c,:)=CLF_Train_output(c).Abstract_level_output;
  %Train_rank_output(c,:,:)=CLF_Train_output(c).Rank_level_output;
  Train_DP_output(c,:,:)=transpose(CLF_Train_output(c).Measurment_level_output);
  %Train_Accuracy(c)=CLF_Train_output(c).Train_Recognition_rate;
  Confusion_Matrx(:,:,c)=CLF_Train_output(c).ConfusionMatrix;
  Test_abstract_output(c,:)=transpose(CLF_Test_output(c).Abstract_level_output);
  Test_rank_output(c,:,:)=transpose(CLF_Test_output(c).Rank_level_output);
  Test_DP_output(c,:,:)=transpose(CLF_Test_output(c).Measurment_level_output);
end
Ensemble_decision=zeros(1,N_test);
%% Now combine the classifiers' output !
switch (CombinitionMethod)
  case {1} % Majority Vote
    Ensemble_decision=mode(Test_abstract_output,1);
  case {2} % Maximum (MAX)
    [Max,Ensemble_decision3d]=max(max(Test_DP_output,[],1),[],2);
    Ensemble_decision=reshape(Ensemble_decision3d,1,N_test);
  case {3} % Sum (SUM)
    [SUM,Ensemble_decision3d]=max(sum(Test_DP_output,1),[],2);
    Ensemble_decision=reshape(Ensemble_decision3d,1,N_test);
  case {4} % % Minimum (MIN)
    [MIN,Ensemble_decision3d]=max(min(Test_DP_output,[],1),[],2);
    Ensemble_decision=reshape(Ensemble_decision3d,1,N_test);
  case {5} % Average (AVR)
    [AVR,Ensemble_decision3d]=max(mean(Test_DP_output,1),[],2);
    Ensemble_decision=reshape(Ensemble_decision3d,1,N_test);
  case {6} % Product (PRO)
    [PRO,Ensemble_decision3d]=max(prod(Test_DP_output,1),[],2);
    Ensemble_decision=reshape(Ensemble_decision3d,1,N_test);
  case {7} % Bayes
    Concatenated_CM=reshape(Confusion_Matrx,N_class,N_class*N_classifiers);
    Concatenated_LM=Concatenated_CM ./ repmat(sum(Concatenated_CM,1),N_class,1);
    index=repmat(0:N_class:(N_classifiers-1)*N_class,N_test,1);
    index=index'+Test_abstract_output;
    D=Concatenated_LM(:,index);
    mu_elements=reshape(D,N_class,N_classifiers,N_test);
    mu=prod(mu_elements,2);
    [Max, Ensemble_decision]=max(mu);
    Ensemble_decision=reshape(Ensemble_decision,1,N_test);
  case {8} % Decision Template
    DT=zeros(N_classifiers,N_class,N_class);
    for c=1:N_class
      class_samples=find(TrainTargets==c);
      DP_class=Train_DP_output(:,:,class_samples);
      DT(:,:,c)=mean(DP_class,3);
    end
    Test_DP_output_replications=repmat(Test_DP_output,[1,1,1,N_class]);
    DT2=reshape(DT,[N_classifiers,N_class,1,N_class]);
    DT3=repmat(DT2,[1,1,N_test,1]);
    sim=(DT3-Test_DP_output_replications).^2;
    mu1=sum(sim,1); mu2=sum(mu1,2);
    [Temp,Ensemble_decision]=min(reshape(mu2,N_test,N_class),[],2);
    Ensemble_decision=Ensemble_decision';
    % The top 7 commands can be replaced wtih the below commands
    %     for j=1:N_test
    %       for c=1:N_class
    %         similarity=(DT(:,:,c)-Test_DP_output(:,:,j)).^2;
    %         mu(c)=sum(similarity(:));
    %       end
    %       [temp,Ensemble_decision(j)]=min(mu);
    %     end
  case {9}  % Dempster-Sahfer fusion; based on Combining pattern classifiers(Kuncheva 2000)
    DT=zeros(N_classifiers,N_class,N_class);
    for c=1:N_class
      class_samples=find(TrainTargets==c);
      DP_class=Train_DP_output(:,:,class_samples);
      DT(:,:,c)=mean(DP_class,3);
    end
    for t=1:N_test
      Current_Test_DP_output=Test_DP_output(:,:,t);
      for j=1:N_class
        Current_DT=DT(:,:,j);
        for i=1:N_classifiers
          Sai(j,i)=1+sum((Current_Test_DP_output(i,:)-Current_DT(i,:)).^2,2);
        end
      end
      Sai=1./Sai;
      Sai=Sai./repmat(sum(Sai,1),N_class,1);
      for j=1:N_class
        for i=1:N_classifiers
          A=prod(1-Sai(:,i))/(1-Sai(j,i));
          b(j,i)=Sai(j,i)*A/(1-Sai(j,i)*(1-A));
        end
      end
      mu=prod(b,2);
      [tmp,Ensemble_decision(t)]=max(mu);
    end
  case {9} % Behavior Knowledge Space (BKS)
    % This method is not written efficinetly at all :(
    N_combinition=N_class^N_classifiers;
    BKS_Table=zeros(N_combinition,N_class+2);
    [combinitionsSerial,combinitions]=MakeBKScombinitions(N_class,N_classifiers);
    for t=1:N_combinition
      Comb=combinitions(t,:);
      samples=[];
      for L=1:N_classifiers
        tmp=(Comb(L));
        CLF_ind=Train_CM_ind{L};
        samples=[samples, cat(2,CLF_ind{:,tmp})]; %samples that has this Comb
      end
      nbar=max(samples)-min(samples)+1;
      Common=hist(samples,nbar);
      CommonLabels=find(Common==N_classifiers) + min(samples)-1 ;
      for c=1:N_class
        BKS_Table(t,c)=length(find(TrainGroup(CommonLabels)==c));
      end
    end
    [BKS_Table(:,N_class+1),BKS_Table(:,N_class+2)]=max(BKS_Table(:,1:3),[],2);
    L=N_classifiers;
    power=10.^(L-1:-1:0);
    power=repmat(power,N_test,1);
    abstract_output_serial=sum(Test_abstract_output'.*power,2);
    for j=1:N_test
      temp=find(combinitionsSerial==abstract_output_serial(j));
      if BKS_Table(temp,N_class+1)~= 0
        Ensemble_decision(j)=BKS_Table(temp,N_class+2);
      else
        Ensemble_decision(j)=mode(Test_abstract_output(:,j));
      end
    end
end
end
function [combinitionsSerial,combinitions]=MakeBKScombinitions(N_class,N_classifiers)
N_vec=repmat(N_class,1,N_classifiers);
N_combinition=prod(N_vec);
N_col=length(N_vec);
combinitions=zeros(N_combinition,N_col);
for ii=1:1:N_col
  if ii==1
    combinitions(:,1)=kron(ones(prod(N_vec(2:end)),1),(1:N_vec(1)).');
  elseif ii==N_col
    combinitions(:,ii)=kron((1:N_vec(ii)).',ones(prod(N_vec(1:ii-1)),1));
  else
    combinitions(:,ii)=kron(ones(prod(N_vec(ii+1:end)),1),kron((1:N_vec(ii)).',ones(prod(N_vec(1:ii-1)),1)));
  end
end
% Change to serial format; i.e. [ 1 3 4] converted to 134
L=N_classifiers;
power=10.^(L-1:-1:0);
power=repmat(power,N_combinition,1);
combinitionsSerial=sum(combinitions.*power,2);
end