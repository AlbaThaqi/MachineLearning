
# Titulli :   Parashikimi i Reshjeve në Shqipëri me Machine Learning

**Lënda**:  Machine Learning

**Profesoresha**: Lule Ahmeti 

**Asistenti i lëndës:** Mërgim Hoti 

**Studimet**: Master - Semestri II

**Drejtimi** : Inxhinieri Kompjuterike dhe Softuerike -  IKS 

**Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK

**Universititeti** :Universiteti i Prishtinës - "Hasan Prishtina"

## Anëtarët e grupit:

- Alba Thaqi
  
- Blerta Krasniqi
  
- Lirak Xhelili


## Përshkrimi i dataset-it

Ky dataset përmban të dhëna historike mbi reshjet në Shqipëri, të organizuara sipas zonave administrative. Ai mund të përdoret për analizë klimatike, parashikime të reshjeve dhe studime mbi modelet e motit.

**Karakteristikat kryesore të datasetit:**

- Numri i rreshtave: 558,091
- Numri i kolonave: 14
- Tipet e të dhënave:
- ![image](https://github.com/user-attachments/assets/5df4274f-e492-4bdf-a8bc-378e6a67ac47)

- Kualiteti i të dhënave:
- Numri i të dhënave të plota:
- ![image](https://github.com/user-attachments/assets/9d2ba501-c42b-45f2-b3f4-78683b210efd)

- Numri i të dhënave null:
- ![image](https://github.com/user-attachments/assets/93a54b9b-0d5b-4bf7-835f-4eaef8102564)


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
  - ![image](https://github.com/user-attachments/assets/83907115-3e8b-4b76-815f-1b753e21dd7b)

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
   - 1. adm2_id (identifikues i regjioneve administrative)
     2. ADM2_PCODE (kodi për identifikimin e regjioneve administrative)
     3. year_month (një periudh e derivuar nga kolona *date*)
        ![image](https://github.com/user-attachments/assets/88148fc1-f5a9-4d70-87d7-d87ef55be5ff)
    Ky proces është kryer për të thjeshtuar analizën e datasetit dhe për të reduktuar të dhënat e tepërta dhe të panevojshme.
    Kolonat *adm2_id* dhe *ADM2_PCODE* përmbajnë vlera për të njejtin informacion, kështu që është më efikase të bashkohen në një kolonë të vetme.

   ## Mostrimi
   ## Detektimi dhe menaxhimi i outliers
   ## Standardizimi i të dhënave
   
   


##  Kerkesat e ktinve me i kshyr ren edhe me i rujt 
Titullin e projektit;
▪ Universiteti, fakulteti, niveli i studimeve, lënda në të cilën ndjeken leksionet dhe
mësimdhënësit e lëndës në të cilën jeni të ngarkuar të punoni projektin;
▪ Emrat e studentëve që kanë kontribuar në projekti;
▪ Të definohet gjithmonë se për cilën fazë po paraqisni rezultatet e fituara, p.sh..:
Faza I: Përshkruaj detyrat që janë të definuara (p.sh..: Përgatitja e modelit);
▪ Përshkruaj detajet e datasetit tuaj, numrin e atributeve, objekteve, madhësinë e datasetit,
burimi nga është marrë, etj.
▪ Shfaq rezultate të gjeneruara përgjatë kësaj faze;


