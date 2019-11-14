%% Deep learning for Job classification
%  191113 Jung Seungwon

clear;
close all;

%% Input

filename = "data.csv";
data = readtable(filename,'TextType','string'); 
head(data)

idx = strlength(data.Category)== 0;
data(idx,:) = [];

data.Category = categorical(data.Category);
figure
h = histogram(data.Category);
xlabel("Category")
ylabel("Frequency")
title("Category Distribution")

classCounts = h.BinCounts;
classNames = h.Categories;

idxLowCounts = classCounts < 10;
infrequentClasses = classNames(idxLowCounts);
idxInfrequent = ismember(data.Category,infrequentClasses);
data(idxInfrequent,:) = [];

cvp = cvpartition(data.Category,'Holdout',0.1);
dataTrain = data(cvp.training,:);
dataTest = data(cvp.test,:);

textDataTrain = dataTrain.Resume;
textDataTest = dataTest.Resume;
YTrain = dataTrain.Category;
YTest = dataTest.Category;

documents = preprocessResume(textDataTrain);
documents(1:5)

bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];

XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear')
documentsTest = preprocessResume(textDataTest);
XTest = encode(bag,documentsTest);
YPred = predict(mdl,XTest);
acc = sum(YPred == YTest)/numel(YTest)

str = ["John H. Smith, P.H.R. 800-991-5187 PO Box 1673  Callahan, FL 32011  info@greatresumesfast.com Approachable innovator with a passion for Human Resources. SENIOR HUMAN RESOURCES PROFESSIONAL Personable, analytical, flexible Senior HR Professional with multifaceted expertise. Seasoned Benefits Administrator with extensive experience working with highly paid professionals in clientrelationship-based settings. Dynamic team leader capable of analyzing alternatives and identifying tough choices while communicating the total value of benefit and compensation packages to senior level executives and employees. CORE COMPETENCIES Benefits Administration – Customer Service – Cost Control – Recruiting – Acquisition Management – Compliance Reporting Retention – Professional Services – Domestic & International Benefits – Collaboration – Adaptability – Change Management Defined Contribution Plans – Auditing – Negotiation – Corporate HR Policies – Full Lifecycle Training – 401(k) – Form 5500 Confidential Files – EEO-1 – AAP – FMLA – STD – LTD – H1-B Visa – Vets 100 – EAP – Processing Payroll HR TECHNOLOGY HRIS Data Management & Auditing – Ultipro Back Office – Ultipro Web Connect Deltek Costpoint – Deltek GCS Premiere – Cognos – ADP Professional Experience HUMAN SERVICES, INC. – Tampa, FL 2010–Present Providing institutional behavioral health and medical management contracting services | 2,500 employees Benefits Manager"];
documentsNew = preprocessResume(str);
XNew = encode(bag,documentsNew);
labelsNew = predict(mdl,XNew)


