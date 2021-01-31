****Deep Learning****

****Super Resolution****

********

****Ognjen Bjeletić ****20****18/****0****447****

****Mihailo Pačarić ****20****18/****0****609****

Opis Problema
-------------

Cilj projekta je da napravimo neuralnu mrezu koja će da poveća
rezoluciju slike 2 puta. Inspiracija je došla od Nvidia grafičkih karti
RTX serije koje podržavaju DLSS (Deep Learning Super Sampling)
tehnologiju. DLSS je tehnologija koju je kompanija Nvidia razvila jer
rezolucije monitora postaju sve veće i igrice postaju zahtevnije za
grafička izračunavanja, ta tehnologija je sposobna da poveća sliku kao i
da uradi Anti Aliasing na njoj što omogućava da grafički procesor računa
sliku na manjoj rezoluciji i da onda DLSS mreza uveća tu sliku i održi
oštrinu slike. Ovo je manje komputaciono zahtevno nego da se svaki okvir
računa na većoj rezoluciji.

Kriterijumske funkcije
----------------------

##### Mean Squared Error

Ova kriterijumska funkcija je najjednostavnija. Vizuelno, rezultati koje
smo dobijali ne izgledaju loše, ali ova kriterijumska funkcija (kao i
prve dve naredne) kao najveći problem ima to što ne uzima u obzir
strukturu slike, već gleda izolovan piksel.

##### Mean **Absolute** Error

Za ovakav problem ovakva kriteriumska funkcija ne pomaže jer slicno kao
MSE odobrava mućenje slike umesto njenog uoštravanja.

##### PSNR

U radovima koje smo gledali, bila je pomenuta ova kriterijumska funkcija
ali nama rezultujuće slike izgledaju mnogo mutnije nego što bi smo
želeli.

##### 

##### VGG

Ova kriteriumska funkcija je jako dobra zato što mrežu uči strukturu
slike. VGGnet je neuralna mreza koja je istrenirana da radi detekciju
objekata. Mi koristimo izlaz VGGnet mreze pre poslednja tri ‘Fully
Connected’ sloja (tada je mreža dekomponovala ulaznu sliku na
komponente) i gledamo srednju kvadratnu grešku izmedju izlaza dobijene i
prave slike. Ova kriterijumska funkcija ima neke fragmente koji se mogu
videti na slici.

##### 

##### VGG Style

Ovde za razliku od VGG kriterijumske gledamo izlaz na svakom
pojedinačnom izlazu pre max pooling sloja i to poredimo. Ova funkcija
ima neke (malo drugačije) fragmente.

##### 

##### 

##### 

##### 

##### 

##### SSIM

SSIM je ispao odličan za naše potrebe. To je kriterijumska funkcija koja
ceni strukturnu sličnost izmedju ulazne i izlazne slike i time uči našu
mrezu da bolje reprodukuje izgubljene informacije. Ova kriterijumska
funkcija ne gleda samo po jedan izolovan piksel, već uzima u obzir i
njegove susede.

##### 

##### SRGAN

SRGAN je GAN primenjen na problem super rezolucije. GAN (Generativna
Kontradiktorna Mreža) je klasa framework-a za mašinsko učenje. Zasniva
se na ideji Zero-Sum igre gde je gubitak jednog agenta dobitak drugog.
Ovde se to implementira sa dve neuralne mreže gde jedna povećava
rezoluciju slike (generator), a druga uči da razlikuje pravu sliku
visoke rezolucije i sliku koja je produkt super
rezolucije(diskriminator). Generator pokušava da nauči kako da “nadmaši”
diskriminatora i tako se poboljšava kvalitet slike. Uz to se koristi i
neka od gore pomenutih kriterijumskih funkcija kao glavni deo
ocenjivanja (mi smo koristili SSIM i VGG Style).

Arhitekture mreze
-----------------

##### Obicna Konvoluciona Mreža

Na ulazu sliku propustimo kroz neki broj konvolucionih slojeva, nakon
čega dupliramo veličinu izlaza algoritmom najbližeg komšije. Nakon toga
slika se pušta kroz još konvolucionih slojeva i onda se formira izlazna
slika.

Posto smo mi trenirali mrezu na slikama velicine 48x48 ova mreza nam je
davala prihvatljive rezultate sa SSIM kriterijumskom i brze je trenirala
od kasnije pomenutih komplikovanijih arhitektura. Ovde se na grafiku
moze videti pretreniranje, to smo na ostalim graficima izbegli
detekcijom pretreniranja.

##### U-net

Ovde se koristi U-oblik (tu se slika prvo smanjuje pooling-om i onda
povećava nekim algoritmom (npr. najbliži komšija), i koriste se veze za
preskakanje (skip connections).

Veze za preskakanje koje smo koristili su slicne vezama koje se nalaze u
ResNet mreži u rezidualnim blokovima, izlaz pre prvog pooling-a dodajemo
na dubinu nakon prvog povećanja. Ovo omogućava da mreža nekad ide
alternativnom putanjom i “preskoči” donji deo U oblika (deo nakon
pooling-a). Razlika u odnosu na ResNet je sto se umesto sabiranja
sadrzaja slojeva, slojevi nadodaju na dubinu (To se koristi u DenseNet
mrezi).

##### 

##### Pixel Shuffle

Pixel Shuffling nije model mreže ali je svakako bitan za pomenuti. To je
sloj koji sluzi za povećanje slike. Umesto povećanja slike algoritmima
za upscaling, slika se povećava tako što se po 4 sloja u dubinu saberu u
jedan veći. Ovde se javlja problem šahovske table ako se mreza ne
trenira dovoljno.

##### 

##### 

##### 

##### 

##### SRGAN

Ovde mozemo pričati arhitekturama dva potrebna modela. Generator je
model koji povećava sliku i za njegovu arhitekturu mozemo koristiti bilo
koju od prethodno opisanih modela(mi smo koristili običnu konvolucionu).
Diskriminator smo napravili sa konvolucionim i pooling slojevima nakon
kojih ide jedan Fully-Connected sloj koji nam daje rezultat. Ova
arhitektura nam je dala najbolje rezultate.
