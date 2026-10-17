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
  case {10} % Behavior Knowledge Space (BKS)
    % Match observed patterns directly: avoid an exponential N_class^N_classifiers table.
    trainPatterns = zeros(N_train,N_classifiers);
    for c=1:N_classifiers
      trainPatterns(:,c) = CLF_Train_output(c).Abstract_level_output(:);
    end
    for t=1:N_test
      matches = all(bsxfun(@eq,trainPatterns,Test_abstract_output(:,t).'),2);
      if any(matches)
        Ensemble_decision(t) = mode(TrainTargets(matches));
      else
        Ensemble_decision(t) = mode(Test_abstract_output(:,t));
      end
    end
  otherwise
    error('Research:UnknownCombination', 'Combination method must be 1 through 10.');
end
end
