
# Projekti :   Parashikimi i Reshjeve të shiut në Shqipëri me Machine Learning

**Lënda**:  Machine Learning

**Profesoresha e lëndës**: Lule Ahmeti 

**Asistenti i lëndës:** Mërgim Hoti 

**Studimet**: Master - Semestri II

**Universititeti** : Universiteti i Prishtinës - " Hasan Prishtina "

**Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK

**Drejtimi** : Inxhinieri Kompjuterike dhe Softuerike -  IKS 


## Anëtarët e grupit:

- Alba Thaqi
  
- Blerta Krasniqi
  
- Lirak Xhelili


## Përshkrimi i dataset-it

Ky dataset përmban të dhëna historike mbi reshjet në Shqipëri, të organizuara sipas zonave administrative. Ai mund të përdoret për analizë klimatike, parashikime të reshjeve dhe studime mbi modelet e motit.
Dataseti është marrë nga website: https://data.humdata.org/, që është pjesë e platformës Humanitarian Data Exchange (HDX). Kjo platformë menagjohet nga United Nations Office for the Coordination of Humanitarian Affairs (OCHA) dhe ka për qëllim të bëjë të lehtë qasjen në të dhëna për organizatat, ose hulumtuesit e ndryshëm në internet.

**Karakteristikat kryesore të datasetit:**

- Numri i rreshtave: 558,091
- Numri i kolonave: 14
- Tipet e të dhënave:
  
  ![image](https://github.com/user-attachments/assets/5df4274f-e492-4bdf-a8bc-378e6a67ac47)


- Numri i të dhënave të plota:
  
 ![image](https://github.com/user-attachments/assets/9d2ba501-c42b-45f2-b3f4-78683b210efd)

- Numri i të dhënave null:
  
  ![image](https://github.com/user-attachments/assets/93a54b9b-0d5b-4bf7-835f-4eaef8102564)


Të dhëna të organizuara sipas njësive administrative

Masa të reshjeve të regjistruara në intervale të ndryshme kohore (orë, ditë)

Përqindja e pikselëve të mbuluar me reshje në rajone specifike


**Kolonat kryesore:**

**date** – Data e regjistrimit të reshjeve

**adm2** – Zona administrative (qyteti, rrethi)

**n_pixels** – Numri i pikselëve të mbuluar me reshje në imazhet satelitore

**rfh** – Sasia e reshjeve në orën e fundit

**r1h, r1h_avg, r1q** – Reshjet dhe mesatarja e reshjeve për 1 orë

**r3h, r3h_avg, r3q** – Reshjet dhe mesatarja e reshjeve për 3 orë

# Faza e parë
## Kërkesat për fazën e parë

# 1. Përgatitja e modelit

  ## Ngarkimi i dataset-it
  - Ngarkimi i dataset-it është realizuar duke përdorur librarinë *pandas*. Përmes funksionit *.head()*, printohen pesë rreshtat e parë të datasetit së bashku me të dhënat.
    
   ![image](https://github.com/user-attachments/assets/83907115-3e8b-4b76-815f-1b753e21dd7b)

   ## Strategjia e trajtimit të vlerave të zbrazëta
   -Duke përdorur funksionin *isnull().sum()*, si rezultatet kthehet numri i vlerave që mungojnë në dataset. Imazhi më poshtë tregon rezultatin e kthyer nga programi për datasetin e zgjedhur. Meqë numri i vlerave që mungojnë është i madh, duhet të përdoren strategji për t'i trajtuar vlerat e zbrazëta.
   
   ![image](https://github.com/user-attachments/assets/37f53e17-d5fb-4f3e-bbf5-e197e4cd7a8b)
   
   - Në këtë dataset janë përdorur disa strategji për trajtimin e vlerave të zbrazëta, duke përfshirë:
   - 1. Zëvendësimi i vlerave me 'None'.
        
  ![image](https://github.com/user-attachments/assets/53015f57-fd40-44c1-908d-18c9eaf7f53a)
        
  - 2. Mbushja e vlerave të zbrazëta me medianën, për kolonat numerike.
       
   ![image](https://github.com/user-attachments/assets/7594053d-96c8-4c18-a04b-4ec3a7daef65)
  
  - 3. Heqja e kolonave me funksionin *drop()*. Kjo strategji është përdorur te kolona 'date', për rreshtin e parë, sepse ka pasur të dhëna jo të duhura për kolonën date, pasi që është bërë formatimi i datës.
    
    ![image](https://github.com/user-attachments/assets/01e8d860-005c-4432-b5b7-da9038d15131)

   ## Agregimi
   - Është realizuar duke kombinuar 3 kolona:
     1. adm2_id (identifikues i regjioneve administrative)
     2. ADM2_PCODE (kodi për identifikimin e regjioneve administrative)
     3. year_month (një periudh e derivuar nga kolona *date*)
        
     ![image](https://github.com/user-attachments/assets/88148fc1-f5a9-4d70-87d7-d87ef55be5ff)
        
    Ky proces është kryer për të thjeshtuar analizën e datasetit dhe për të reduktuar të dhënat e tepërta dhe të panevojshme.
    Kolonat *adm2_id* dhe *ADM2_PCODE* përmbajnë vlera për të njejtin informacion, kështu që është më efikase të bashkohen në një kolonë të vetme.

   ## Detektimi dhe menaxhimi i outliers
   - Është realizuar duke përdorur Z-score, që identifikon data points, që devijojnë nga mesatarja. Një absolut e lartë e Z-score nënkupton që ka outlier. Zakonisht përdoret një kufi prej 3 të Z-score dhe të       gjitha vlerat mbi të llogariten si outlier.
     
     ![image](https://github.com/user-attachments/assets/c1568048-9bdb-4e69-a0d5-16a7b8b23b27)
     
   - Për t'i trajtuar të dhënat është përdorur metoda *capping/flooring*, për të reduktuar ndikimin e tyre pa i hequr komplet nga dataseti.
     
     ![image](https://github.com/user-attachments/assets/d11ede13-03cf-4f0a-a67c-1760ca5775be)


   ## Standardizimi i të dhënave
   - Ky proces është realizuar për të siguruar që të dhënat numerike janë në një shkallë të njejtë. Është përdorur funksioni *StandardScaler()*, që i transformon të dhënat, ashtu që secila kategori të ketë një mesatare 0 dhe një devijim standard 1.
     
     ![image](https://github.com/user-attachments/assets/1ee6bb38-f213-498e-89af-780332d5984f

     
# Faza e dytë

## Faza II: Analiza dhe evaluimi (ritrajnimi)
Pas rezulateve të arritura në  fazën e parë  normalizimi, pastrimi i të  dhënave, binarimizimi dhe standardizimin e të dhënave.
Kalojmë tek pjesa e trajnimit të datasetit e cila bazohet në disa kërkesa:

Caktimi i 4 algoritmeve - 2 supervised (Decision Tree, Random Forest) + 2 unsupervised (Agglomerative Clustering, Spectral Clustering);	

Aplikimi i metrikave të performancës së algoritmeve: Accuracy, Precision, Recall and F1 Score.

Vizualimi dhe diskutimet e dokumentuara të rezultateve duke krahasuar algoritmet ndërmjet tyre, arsyetimi i performancës kundrejt të dhënave të preprocesuara në dataset.

# Trajnimi i modelit - datasetit
Fillon me krijimin e një kolonë të re të quajtur rain_label, e cila përdoret si target për modelet klasifikuese.

![image](https://github.com/user-attachments/assets/d0e364e3-2e36-45fc-8061-212508b26983)


Një pjesë që nuk duhet të injorohet është pjesa nëse dataseti është i balancuar apo i pabalancuar.

![image](https://github.com/user-attachments/assets/ac308931-cc2e-4133-aa81-94124fc7da51)


Në rastin tonë dataseti ynë ka kaluar gjitha proceset e fazës së parë me sukses dhe është një dataset i balancuar.

Rain label distribution:

0      282512

1      275578

![image](https://github.com/user-attachments/assets/7a6689ad-5b8d-4907-a0a5-5c223dcd021d)

Pasi të  krijohet një kopje e të gjitha kolonave numerike në X, e cila përfaqëson veçoritë (features) që do i jepen modelit. 
Kolona rfh hiqet sepse është përdorur për të krijuar targetin rain_label dhe do të krijonte një "leakage" nëse lihet brenda. 
Më pas, variabla y vendoset të jetë rain_label, që tregon nëse reshjet në një ditë të caktuar janë të larta apo jo. 
Në fund, të dhënat ndahen në dy pjesë: X_train dhe X_test për trajnim dhe testim, duke përdorur train_test_split me 70% për trajnim dhe 30% për testim, 
dhe duke përdorur stratify=y për të ruajtur shpërndarjen e barabartë të klasave në të dy ndarjet. 

# --- Algorimet Supervised  ---

 **Decision Tree** dhe **Random Forest** janë përdorur si algoritme klasifikimi për shkak të efikasitetit të tyre me të dhëna numerike dhe kapacitetit për të interpretuar rezultatet. 
 Decision Tree është i lehtë për t’u vizualizuar dhe kuptuar, duke u bazuar në ndarje të thjeshta të të dhënave.

 Në këtë pjesë të kodit, krijohet një model DecisionTreeClassifier me thellësi maksimale të kufizuar në 5 (max_depth=5) për të shmangur mbingarkimin (overfitting). Kjo pemë vendimmarrjeje trajnohet mbi të dhënat e trajnimit X_train dhe y_train për të mësuar rregullat që ndajnë ditët me reshje të ulëta dhe të larta në bazë të veçorive meteorologjike.
 
 ![image](https://github.com/user-attachments/assets/414444c2-0c8d-4f18-a61d-6c5e6f5a842a)

 Rezultatet në datasetin tonë:
 
![image](https://github.com/user-attachments/assets/ecd6de08-ed2b-4781-9a8e-61a33998b6bf)

 Random Forest ka avantazhin e performancës më të mirë dhe rezistencës ndaj overfitting.
 Në këtë pjesë trajnohet një model RandomForestClassifier me 100 pemë dhe thellësi maksimale të kufizuar në 5 
 për të parandaluar overfitting. Ky model kombinon vendimet e shumë pemëve për të arritur një klasifikim më të saktë
 dhe të qëndrueshëm mbi reshjet, duke u bazuar në veçoritë meteorologjike.
 
![image](https://github.com/user-attachments/assets/9afb82cf-823f-4c51-a9db-d795c0097180)

 Rezultatet në datasetin tonë:
 
![image](https://github.com/user-attachments/assets/e86b17a4-fdbe-44b2-bccc-c148ec8ac70d)
                      
 Të dy janë të përshtatshëm për dataset-in e reshjeve sepse përballojnë mirë outliers dhe përzierje tiparesh me shkallë të ndryshme.
 
![image](https://github.com/user-attachments/assets/f358ee28-c133-4611-b272-e02cf5db3154)

Ky bllok kodi përllogarit metrikat kryesore të klasifikimit (accuracy, precision, recall, F1-score) dhe, nëse kërkohet,
vizaton matricën e konfuzionit për të analizuar rezultatet e parashikimeve. Gjithashtu, përdoret validimi i kryqëzuar 
me 5 folds për të vlerësuar qëndrueshmërinë e modeleve në të gjithë dataset-in.

![image](https://github.com/user-attachments/assets/ff6757a3-4e14-4c39-b0a0-d76577fa7de9)

# -- Algorimet Unsupervised--

Dy algoritme të clustering-ut për të analizuar sjelljen e reshjeve në mënyrë të paetiketuar (unsupervised): Agglomerative Clustering, i cili bazohet në ndarje hierarkike duke bashkuar instancat më të ngjashme, dhe Spectral Clustering, që përdor informacionin e grafit për të identifikuar ndarje komplekse në të dhëna. 

Të dy algoritmet janë aplikuar mbi një mostër të të dhënave të reduktuara me PCA dhe rezultatet janë krahasuar vizualisht 
dhe me metrikën ARI për të vlerësuar sa mirë korrespondojnë me targetin rain_label.
Të dyja këto qasje ndihmojnë në vlerësimin e strukturës latente të reshjeve pa pasur nevojë për klasifikim të drejtpërdrejtë.

Agglomerative Clustering për të grupuar të dhënat në dy klasë pa përdorur etiketa. Ai fillon me çdo pikë si një cluster më vete dhe bashkon gradualisht pikët më të afërta. Etiketat e krijuara krahasohen me klasat reale (rain_label) përmes metrikave dhe ARI për të vlerësuar sa mirë janë grupuar të dhënat. Gjithashtu matet dhe koha e ekzekutimit të algoritmit.

![image](https://github.com/user-attachments/assets/ecd27aa4-828b-410b-9c96-0747a927508b)

Rezultatet në datasetin tonë:
 
![image](https://github.com/user-attachments/assets/f235e6b6-cbf5-487d-9323-1eeaf7aba0bd)

Spectral Clustering për të grupuar të dhënat në dy klasë bazuar në afërsinë midis pikave, duke përdorur grafin e fqinjësisë më të afërt (affinity='nearest_neighbors'). Algoritmi përpunon një mostër të të dhënave (X_sample), llogarit etiketat e klasave përmes fit_predict, dhe më pas vlerëson performancën përmes metrikave si accuracy, precision, recall, F1-score dhe ARI (Adjusted Rand Index), së bashku me kohën totale të përpunimit.
 
![image](https://github.com/user-attachments/assets/57c1fa70-637a-4150-9398-55c6b07897e4)

Rezultatet në datasetin tonë:

![image](https://github.com/user-attachments/assets/43ed06fa-518b-44f0-8c4e-5a90136ed137)


# Vizualizimi 

## 1. Shpërndarja e Etiketave për Reshje
Ky grafik tregon shpërndarjen e të dhënave në dy klasa: "Pak reshje" dhe "Shumë reshje". Siç shihet, të dhënat janë të balancuara pothuajse në mënyrë të barabartë: rreth 50.6% për “Pak reshje” dhe 49.4% për “Shumë reshje”. Ky lloj shpërndarjeje është shumë i favorshëm për trajnimin e modeleve mbikëqyrëse, pasi redukton rrezikun e paragjykimit të modelit ndaj njërës klasë.
![image](https://github.com/user-attachments/assets/33873f41-4efb-483b-b53c-ea145bd06e79)

## 2. Krahësimi i Performancës së Modeleve Mbikëqyrëse
Në këtë grafik shtyllor janë paraqitur metrikat e performancës për dy modele klasifikimi: Decision Tree dhe Random Forest. Për secilin model janë krahasuar metrikat:
- Saktësia (Accuracy)
- Precision
- Recall
- F1-Score
- CV Accuracy (saktësia mesatare në validimin me kryqëzim)
- Rezultatet tregojnë se të dy modelet kanë performancë shumë të ngjashme, me vlera shumë të larta në të gjitha metrikat, çka sugjeron se të dy modelet janë të aftë të bëjnë parashikime të sakta mbi të dhënat e reshjeve.
![image](https://github.com/user-attachments/assets/5ce4589d-a2a4-4fc7-91f6-523ace3f449d)

## 3. Agglomerative Clustering me PCA dhe Spectral Clustering me PCA
Ky grafik përfaqëson rezultatet e grupimit të të dhënave duke përdorur algoritmin Agglomerative Clustering, dhe është vizualizuar pas reduktimit të dimensioneve me PCA (Principal Component Analysis) në dy komponentë kryesorë. Çdo pikë përfaqëson një shembull nga të dhënat, dhe ngjyra e saj i korrespondon një grupi (cluster). Shihen dy grupe të dallueshme, që tregon se të dhënat kanë një ndarje të natyrshme sipas karakteristikave të tyre.
Ngjashëm me grafikën e mëparshëm, ky vizualizim tregon rezultatet e Spectral Clustering të aplikuar mbi të njëjtat të dhëna dhe të reduktuara me PCA. Edhe këtu, të dhënat ndahen në dy grupe të dallueshme, me një ndarje më të theksuar sesa në Agglomerative Clustering. Kjo tregon se Spectral Clustering ka kapur më mirë strukturën e brendshme të të dhënave.
![image](https://github.com/user-attachments/assets/69232b16-07ad-473f-ab80-2405eb71fc86)


# Faza e tretë
Ritrajnimi i datasetit dhe përdorimi i veglave të ML  
- Pas përpunimit fillestar dhe analizës eksploruese të të dhënave klimatike të reshjeve, është ndërmarrë faza e ritrajnimit me qëllim përmirësimin e performancës së modeleve të Machine Learning. 
Kjo fazë përfshin aplikimin e teknikave të avancuara, si optimizimi i hiperparametrave përmes GridSearchCV, përdorimi i transformimeve logaritmike për trajtimin e shpërndarjeve jo normale dhe normalizimi i plotë i veçorive numerike për të ruajtur qëndrueshmërinë në trajnim.
Kodi i përshtatet fazës së dytë me disa përmisime të rezultateve.

**Optimizimi i Decision Tree me GridSearchCV** - mundohet ti eksploroj kombinime të ndryshme të hiperparametrave për DecisionTreeClassifier.

**max_depth:** thellësia maksimale e pemës.

**min_samples_split:** numri minimal i shembujve që kërkohen për të bërë ndarjen e një nyje.

**min_samples_leaf:** numri minimal i shembujve në një degë fundore.

**criterion:** funksioni i matjes së ndarjes (gini ose entropy).


![image](https://github.com/user-attachments/assets/42887f5d-bce2-4600-856f-10e08bf31ce6)

GridSearchCV për të testuar përdor çdo kombinim në param_grid me validim të kryqëzuar 5-fish (cv=5).

![image](https://github.com/user-attachments/assets/e67b1e12-7d3f-4779-9439-e02769d89a4d)

**Optimizimi i Random Forest me GridSearchCV**

Parametra që eksplorohen për Random Forest:

**n_estimators:** numri i pemëve në pyll.

**max_depth:** thellësia maksimale për secilën pemë.

**min_samples_split:** numri minimal i shembujve që kërkohen për të bërë ndarjen e një nyje.

Trajnon modele me të gjitha kombinimet dhe i teston me cv=5.

![image](https://github.com/user-attachments/assets/f40fe7a7-7546-41d7-a831-a6c7d5985694)

Rreshti i parë merr modelin më të mirë të trajnuar nga GridSearchCV, modelin RandomForestClassifier me kombinimin më të mirë të parametrave (n_estimators, max_depth, min_samples_split, etj.),të cilin GridSearchCV e ka testuar përmes validimit të kryqëzuar (cv=5) dhe e ka zgjedhur në bazë të performancës.

Tek rreshti i dytë -ky model i optimizuar (best_rf) përdoret për të parashikuar klasat (rain_label) për setin e testimit (X_test).Rezultati është një listë me vlera 0 ose 1 (reshje të ulëta apo të larta).

Ndërsa tek rreshti tretë - llogarit Accuracy, Precision, Recall,F1 Score dhe shfaq Confusion Matrix.

![image](https://github.com/user-attachments/assets/23ea0807-4a9f-432d-9742-8d27c7fe370f)

Përcakton një listë me kolona që kanë vlera shumë të ndryshme në madhësi (d.m.th. shpërndarje jo-normale, outliers të mëdhenj).

np.log1p(x) llogarit logaritmin natyror të (1 + x) për secilën vlerë në kolonë.
Ul efektin e outliers që do ndikonin shumë algoritmin (sidomos Decision Tree dhe Random Forest).

Ndërsa pjesa tjetër Merr të gjitha kolonat numerike (ato të përzgjedhura më parë me numerical_columns).
I kalon nëpër StandardScaler i normalizon duke i kthyer në shpërndarje me mesatare = 0 dhe devijim standard = 1.

Eliminon ndikimin e kolonave që kanë vlera më të mëdha vetëm për shkak të njësive të matje
![image](https://github.com/user-attachments/assets/2851d2c7-2ef2-4d3a-8b9e-f33c62d0d8e3)

Për të ndërtuar variablën target për klasifikim binar, u krijua kolona rain_label, ku vlerat e reshjeve orare (rfh) u klasifikuan si 1 nëse ishin mbi medianën dhe 0 ndryshe. Më pas, rfh u përjashtua nga veçoritë hyrëse për të shmangur ndikimin direkt në model. Dataseti u nda në grupe trajnimi dhe testimi në raport 70:30, duke përdorur stratifikim për të ruajtur balancën mes klasave në të dy ndarjet.

![image](https://github.com/user-attachments/assets/915ca14c-b484-40c3-a00d-19fe06c63153)

Për të vlerësuar performancën e modelit bazë dhe atij të optimizuar, u ndërtua një strukturë krahasuese përmes një DataFrame. 
Për secilin model, u llogaritën metrikat kryesore si saktësia (accuracy) dhe F1 score duke përdorur të dhënat e testimit. Këto metrika u ruajtën në formë liste dhe më pas u përfshinë në një tabelë krahasuese (compare_df), që shërben si bazë për analizimin e përmirësimeve pas ritrajnimit dhe për vizualizimin e tyre.

![image](https://github.com/user-attachments/assets/dad9577e-e7a0-4527-a80b-6e69123434dd)

## Veglat dhe aplikimi në ML 

Në rastin tonë kemi përdoru 5 vegla: ydata_profiling, sweetviz, mlflow, wandb dhe joblib.

**YDATA_PROFILING** 

Një vegël për analizë eksploruese të të dhënave (EDA) që gjeneron automatikisht një raport të plotë mbi dataset-in. Përdoret për të kuptuar shpërndarjen e kolonave, mungesat, outliers, korelacionet dhe strukturën e përgjithshme të të dhënave pa shkruar kod analizues manual.Gjeneron një raport interaktiv në format HTML dhe ruhen në rainfall_report.html.

![image](https://github.com/user-attachments/assets/e9ea2fb3-d891-4152-a13b-78921409f74f)


**SWEETVIZ**

Një vegël për krahasim të veçorive ndërmjet dy seteve të të dhënave – zakonisht train dhe test.
Shërben për të kontrolluar nëse ndarja e të dhënave ka ndikuar në strukturën e tyre dhe për të analizuar se si shpërndahen veçoritë krahasuar me targetin.HTML raport me grafikë dhe analiza automatike ruhen në sweetviz_report.html.

![image](https://github.com/user-attachments/assets/eb2b3b88-6136-4e74-b98f-2627380e3c89)


**MLFLOW**

Platformë për menaxhimin e ciklit jetësor të eksperimenteve të Machine Learning.
Regjistron parametrat, metrikat, dhe vetë modelin gjatë çdo eksperimenti. Lejon krahasim ndërmjet konfigurimeve të ndryshme dhe gjeneron një dashboard lokal. Të dhëna për parametrat dhe metrikat e modelit në sistemin Web UI lokal, dhe një strukturë dosjeje .mlruns/ për ruajtje të brendshme ku shfaqet me komandën mlflow ui në http://localhost:5000.

![image](https://github.com/user-attachments/assets/8512b5d5-4053-4755-81ed-2031e6db98d6)


**WANDB**

Një platformë cloud për ndjekjen dhe vizualizimin e eksperimenteve të ML në kohë reale.Vizualizon grafikët e performancës (accuracy, loss, F1, etj), ruan historikun e konfigurimeve, dhe mbështet punën në ekip ku shfaqet në faqen wandb.ai, në projektin e konfiguruar nga përdoruesi.

![image](https://github.com/user-attachments/assets/b5b11095-4437-4655-b9df-4437fc0f9267)


**JOBLIB**

Librari për ruajtjen dhe ngarkimin e objekteve Python, zakonisht përdoret për të ruajtur modele të trajnuara. Lejon që një model i trajnuar të ruhet si skedar binar dhe të rishfrytëzohet më vonë pa e ritrajnuar ku ruhen në best_rf_model.joblib.

![image](https://github.com/user-attachments/assets/769b6804-6aa1-4766-a691-4f5158493500)


## Gjenerimi i file-s
Për të përmbledhur më mirë punën e kryer gjatë këtij projekti, janë gjeneruar dy file-s, që paraqesin një përshkrim të punës, duke filluar nga dataset-i, karakteristikat e dataset-it dhe deri te rezultatet.
Figura e mëposhtme shfaq një përshkrim të përgjithshëm të dataset-it, pasi që të dhënat janë pastruar dhe standardizuar.

![image](https://github.com/user-attachments/assets/f8fb2868-c0c2-47d1-a2fa-84743e846540)

 Nëse kalojmë përgjatë kësaj tabele, shihen edhe marrëdhëniet që kanë vetitë e dataset-it ndërmjet veti.
 
![image](https://github.com/user-attachments/assets/328416fe-b6cc-44be-b9d5-24f669365df6)

Meqë qëllimi i projektit ka qenë të parashikohen reshjet përgjatë muajve dhe viteve, atëherë është dhënë një pamje më e detajuar e kolonës *date* dhe një histogram, që vizualizon shpërndarjen e të dhënave të reshjeve në bazë të kohës (kolona date) nga viti 1981 deri në 2025.

![image](https://github.com/user-attachments/assets/5dfd1fa0-d38c-4b72-8f1d-fae3f4df405f)

Matrica e korrelacionit tregon marrëdhënien ndërmjet karakteristikave të datasetit, duke u paraqitur në intervalin [0-1]. Sa më e lartë të jetë vlera (ose më e theksuar ngjyra), aq më e lartë është marrëdhënia ndërmjet dy karakteristikave, për të ndikuar në rreshje.

![image](https://github.com/user-attachments/assets/aeae8cf6-3fd7-4671-b469-863ade8a0a73)

Vizualizimi i vlerave null nëpër secilën kolonë para trajnimit të datasetit

![image](https://github.com/user-attachments/assets/b4f0c8c0-84d2-4c78-b0b1-3045a0020219)
Analiza e Ndërveprimeve - Ndërveprimet midis variablave në dataset-in e reshjeve. Ndryshimet në reshje janë të dukshme sipas rajoneve (ADM2_PCODE). Ndërveprimet sugjerojnë që modelet e parashikimit duhet të marrin parasysh vendndodhjen gjeografike dhe kohën e vitit.
![image](https://github.com/user-attachments/assets/44f6c12e-5bb3-4023-8e08-4799df7934f2)  

Në këtë figurë janë analizuar kolonat rfq, r1q dhe r3q, të cilat përfaqësojnë sasinë e reshjeve në total dhe për intervalet kohore 1h dhe 3h. Histogramët tregojnë një shpërndarje të pabalancuar, ku shumica e vlerave janë të ulëta, ndërsa disa pika ekstreme paraqesin reshje të mëdha. Kjo pasqyrohet edhe në vlerat e skew që tejkalojnë 1. Për të përmirësuar këtë shpërndarje, në kod janë aplikuar transformime logaritmike (log1p), që shërbejnë për të zvogëluar ndikimin e outliers dhe për të bërë të dhënat më të përshtatshme për modelet e klasifikimit.

![qeto](https://github.com/user-attachments/assets/4e00a3c1-a711-465f-a871-d110ff6edf13)

Kjo figurë tregon analizën e kolonave n_pixels dhe rfh_avg. Kolona n_pixels përmban shumë pak vlera të ndryshme, çka tregon variacion të ulët dhe ndikim potencialisht të vogël në model. Ndërkohë, rfh_avg, që përfaqëson mesataren e reshjeve orare, ka një shpërndarje pak të djathtë, por të pranueshme për modelim pas standardizimit. Këto të dhëna tregojnë që ndërsa disa veçori mund të kenë më shumë informatë, të tjerat mund të kenë nevojë për vlerësim të kujdesshëm për peshën që i jepet gjatë trajnimit.


![qeto1](https://github.com/user-attachments/assets/6abda029-821f-425a-a5d6-0a43368d2841)

Në këtë figurë vërehen shpërndarjet e kolonave që përfaqësojnë reshje në periudha të shkurtra kohore: 1 orë dhe 3 orë, bashkë me mesataret përkatëse. Pas transformimeve të aplikuara në kod, histogramët e train dhe test duken të balancuara dhe mjaft të ngjashme, çka tregon një ndarje të drejtë të dataset-it. r3h_avg veçanërisht ka një shpërndarje shumë afër normales, me skew minimal. Kjo sugjeron që këto veçori janë të përgatitura mirë për modelim dhe ka gjasa të kontribuojnë pozitivisht në performancën e klasifikuesve.

![qeto2](https://github.com/user-attachments/assets/1f1c62f2-7d74-4ce7-961b-60dc20b0a895)

Figura e fundit paraqet sërish kolonat rfq, r1q dhe r3q, duke ofruar një verifikim të mëtejshëm të shpërndarjes së këtyre veçorive pas ndarjes në train dhe test. Distribucionet e të dhënave mbeten të ngjashme mes dy seteve, duke treguar konsistencë dhe mungesë bias-i. Kjo përforcon faktin që ndarja e të dhënave është bërë me stratifikim të saktë dhe se të dhënat janë të përgatitura mirë për trajnimin e mëtejshëm të modeleve të Machine Learning.

![qeto3](https://github.com/user-attachments/assets/52e44ada-fc87-4176-8424-53b4a4711d75)




## Rezultatet 

Pas ritrajnimit të dataset-it duke përdorur teknika si transformimi logaritmik, normalizimi i veçorive dhe optimizimi i hiperparametrave përmes GridSearchCV, u vu re një përmirësim i ndjeshëm në performancën e modelit Random Forest. Saktësia dhe F1-score e modelit të optimizuar u rritën krahasuar me versionin fillestar, duke treguar se përgatitja e kujdesshme e të dhënave dhe përzgjedhja e parametrave adekuatë ndikon drejtpërdrejt në efikasitetin e parashikimeve.

![WhatsApp Image 2025-05-18 at 13 44 14_d89bfe05](https://github.com/user-attachments/assets/e8fbfad4-fdc4-4004-83dc-9fbb81682180)

![WhatsApp Image 2025-05-18 at 13 44 14_3e3f1d2a](https://github.com/user-attachments/assets/fcd25792-e6cd-48f0-98fc-69732ab584b8)



## Diskutime dhe konkluzion 
Në këtë projekt u trajtua një dataset mbi reshjet klimatike, duke kaluar nëpër faza standarde të përgatitjes së të dhënave si pastrimi, transformimi logaritmik, normalizimi dhe ndarja në train/test. Për klasifikim u përdorën algoritmet RandomForestClassifier dhe DecisionTreeClassifier, ku përmirësimi i performancës u arrit përmes GridSearchCV.

Përveç modeleve, janë integruar edhe vegla ndihmëse të Machine Learning si:

**ydata_profiling** për analizë automatike të të dhënave,

**sweetviz** për krahasim vizual të train/test,

**mlflow** dhe **wandb** për regjistrim dhe vizualizim të eksperimenteve,dhe joblib për ruajtje të modelit final.

Integrimi i këtyre veglave rriti transparencën, përsëritshmërinë dhe profesionalizmin e procesit të zhvillimit të modelit. Kjo qasje e plotë nga përgatitja e të dhënave deri te ruajtja e modelit përbën një cikël të plotë të Machine Learning dhe ofron një bazë solide për aplikime të mëtejshme në parashikimin e reshjeve ose fusha të tjera klimatike.










   
   





