
# Projekti :   Parashikimi i Reshjeve të shiut në Shqipëri me Machine Learning

**Lënda**:  Machine Learning

**Profesoresha e lëndës**: Lule Ahmeti 

**Asistenti i lëndës:** Mërgim Hoti 

**Studimet**: Master - Semestri II

**Drejtimi** : Inxhinieri Kompjuterike dhe Softuerike -  IKS 

**Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK

**Universititeti** : Universiteti i Prishtinës - " Hasan Prishtina "

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

Faza II: Analiza dhe evaluimi (ritrajnimi)
Përgjatë kësaj faze duhet të keni të paraqitura të gjitha detajet që keni aplikuar, rezultatet
që keni fituar, jo vetëm të vendosen copy-paste por të përcjellën me sqarime e diskutime të
detajuara. Kjo duke argumentuar dhe arsyetuar pse keni vendos të aplikoni/ përdorni atë
formë të teknikave dhe rezultatet që keni fituar duke i diskutuar;
Trajnimi i modelit
   
   





