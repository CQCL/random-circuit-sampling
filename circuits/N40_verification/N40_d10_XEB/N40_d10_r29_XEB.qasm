OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.457373605059496*pi,1.362063866663953*pi) q[0];
U1q(0.651552705461271*pi,1.324305215112324*pi) q[1];
U1q(0.462075747602203*pi,0.0465334216497648*pi) q[2];
U1q(0.38795801106428*pi,0.421927273476476*pi) q[3];
U1q(0.249165278341374*pi,1.282530894026382*pi) q[4];
U1q(0.610313519950173*pi,0.911813532396616*pi) q[5];
U1q(0.585065455710394*pi,0.9462127242548699*pi) q[6];
U1q(0.301247563215749*pi,1.740414199258488*pi) q[7];
U1q(0.635368052211461*pi,1.24371138602448*pi) q[8];
U1q(0.434299142592174*pi,0.589499285023724*pi) q[9];
U1q(0.491570478010372*pi,1.217881935772744*pi) q[10];
U1q(0.575910110317337*pi,0.516391943148455*pi) q[11];
U1q(0.396905938415123*pi,1.243480312567757*pi) q[12];
U1q(0.264368181452632*pi,0.17760271922584*pi) q[13];
U1q(0.0762913540824814*pi,0.282131941016978*pi) q[14];
U1q(0.37315561404835*pi,0.217889779611329*pi) q[15];
U1q(0.192739295746087*pi,1.7019272395294571*pi) q[16];
U1q(0.464136149249484*pi,1.252705690635413*pi) q[17];
U1q(0.371387304408651*pi,0.61585532823732*pi) q[18];
U1q(0.614935156030656*pi,1.20926482424017*pi) q[19];
U1q(0.564996339829367*pi,0.00837399548281259*pi) q[20];
U1q(0.68950379192682*pi,1.62104268249105*pi) q[21];
U1q(0.725727694583893*pi,1.5073556770110779*pi) q[22];
U1q(0.386158316304247*pi,0.85775060128832*pi) q[23];
U1q(0.379386019525153*pi,1.060750925839367*pi) q[24];
U1q(0.161338467728432*pi,0.299583680887165*pi) q[25];
U1q(0.541879973041732*pi,1.02501180759388*pi) q[26];
U1q(0.636875126946119*pi,0.790056209007503*pi) q[27];
U1q(0.737156407409112*pi,0.861526392983535*pi) q[28];
U1q(0.440728965742691*pi,0.38759592418853006*pi) q[29];
U1q(0.782874960807075*pi,0.211008325039185*pi) q[30];
U1q(0.549381245820525*pi,1.402189510150829*pi) q[31];
U1q(0.534716947095533*pi,1.743268096544941*pi) q[32];
U1q(0.629310108483309*pi,0.547534475207585*pi) q[33];
U1q(0.764710282243097*pi,0.973718594941473*pi) q[34];
U1q(0.229467965109491*pi,0.784201245055115*pi) q[35];
U1q(0.329244076625588*pi,1.7233425151514599*pi) q[36];
U1q(0.718389714898395*pi,0.751650703422015*pi) q[37];
U1q(0.741777276873775*pi,1.719875529664978*pi) q[38];
U1q(0.498527519359623*pi,0.804661056493389*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[11],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[37],q[3];
RZZ(0.5*pi) q[4],q[31];
RZZ(0.5*pi) q[20],q[5];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[38];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[25];
RZZ(0.5*pi) q[10],q[18];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[39],q[15];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[26],q[23];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[27],q[33];
RZZ(0.5*pi) q[34],q[32];
U1q(0.303064365187059*pi,0.44097367269367993*pi) q[0];
U1q(0.676035529184022*pi,0.18980743766372*pi) q[1];
U1q(0.533594655286266*pi,1.4786953099439102*pi) q[2];
U1q(0.474329461034015*pi,0.8135296748229002*pi) q[3];
U1q(0.291021683792484*pi,1.56875044259526*pi) q[4];
U1q(0.531157456253235*pi,0.57081202066982*pi) q[5];
U1q(0.755392406501043*pi,1.3938604961374503*pi) q[6];
U1q(0.732349002257737*pi,0.62499773194808*pi) q[7];
U1q(0.273680146911589*pi,0.03583965638382991*pi) q[8];
U1q(0.645001131625877*pi,1.27989160192451*pi) q[9];
U1q(0.414923549639061*pi,0.010450843076990068*pi) q[10];
U1q(0.400835087289142*pi,0.79781435117667*pi) q[11];
U1q(0.471160847712292*pi,1.24337181141449*pi) q[12];
U1q(0.588869285178662*pi,0.3836127460734202*pi) q[13];
U1q(0.667233178053145*pi,0.1665990969014599*pi) q[14];
U1q(0.430985508973245*pi,0.018157995231329993*pi) q[15];
U1q(0.385632076572787*pi,0.21426957490475007*pi) q[16];
U1q(0.20221303247194*pi,0.5187591241843799*pi) q[17];
U1q(0.638563845596391*pi,0.08448883487649006*pi) q[18];
U1q(0.61741892197607*pi,0.97599704277886*pi) q[19];
U1q(0.322496347122034*pi,0.3208569499985101*pi) q[20];
U1q(0.444799987672631*pi,0.31192714241114006*pi) q[21];
U1q(0.720422995120985*pi,1.60886081984148*pi) q[22];
U1q(0.757060511647927*pi,1.65600965781046*pi) q[23];
U1q(0.139444827697677*pi,1.0301722762135301*pi) q[24];
U1q(0.84543848215483*pi,1.336070778968264*pi) q[25];
U1q(0.310162671091383*pi,0.80794332803261*pi) q[26];
U1q(0.142611335968018*pi,0.5908469526851201*pi) q[27];
U1q(0.437069373031552*pi,1.717421484352381*pi) q[28];
U1q(0.405366002523905*pi,1.06680199947373*pi) q[29];
U1q(0.414611693496284*pi,1.146288522064822*pi) q[30];
U1q(0.382999498799925*pi,1.3790541110098502*pi) q[31];
U1q(0.773342910200159*pi,0.32943575105846*pi) q[32];
U1q(0.367717140931974*pi,1.568946041700012*pi) q[33];
U1q(0.743187536014513*pi,0.31903339454577995*pi) q[34];
U1q(0.874722468498026*pi,1.6307026831812501*pi) q[35];
U1q(0.359131218336163*pi,0.7416707553822*pi) q[36];
U1q(0.195433910978276*pi,0.49254906573087*pi) q[37];
U1q(0.567981471674021*pi,1.7698739087380302*pi) q[38];
U1q(0.579903365626589*pi,1.4623079299495991*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[9],q[31];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[29],q[34];
RZZ(0.5*pi) q[32],q[36];
U1q(0.523664414895558*pi,1.24155883069119*pi) q[0];
U1q(0.334028331760123*pi,1.04010013391974*pi) q[1];
U1q(0.581784607978465*pi,1.8303470784590896*pi) q[2];
U1q(0.54987533345274*pi,1.0693162743052804*pi) q[3];
U1q(0.751507671391208*pi,0.26140502447274017*pi) q[4];
U1q(0.600668728985159*pi,0.6364360251097501*pi) q[5];
U1q(0.143845548326047*pi,0.5441338204411696*pi) q[6];
U1q(0.314316418136146*pi,0.8444141846650304*pi) q[7];
U1q(0.261680845058706*pi,1.3284644899428102*pi) q[8];
U1q(0.877551914132812*pi,1.9429941038026302*pi) q[9];
U1q(0.167357996752165*pi,0.21265849285000016*pi) q[10];
U1q(0.662043024425179*pi,0.055209636220030056*pi) q[11];
U1q(0.732299309596504*pi,0.7258058154553799*pi) q[12];
U1q(0.247302728817245*pi,1.97982082979562*pi) q[13];
U1q(0.779465007737234*pi,0.10284165108410015*pi) q[14];
U1q(0.539040337787043*pi,1.1895727626395898*pi) q[15];
U1q(0.779169344092319*pi,1.8230807733290098*pi) q[16];
U1q(0.477731067361836*pi,1.1671605997091996*pi) q[17];
U1q(0.187114396132525*pi,1.8969676058144804*pi) q[18];
U1q(0.465074362279727*pi,1.6898401855478404*pi) q[19];
U1q(0.336221694385535*pi,1.5926822978985502*pi) q[20];
U1q(0.325583105802168*pi,0.5571282739640298*pi) q[21];
U1q(0.0826387198661232*pi,1.7145484071004402*pi) q[22];
U1q(0.64105770857762*pi,0.15822941237735*pi) q[23];
U1q(0.450174802084126*pi,0.5036944426071899*pi) q[24];
U1q(0.621195504798773*pi,1.7568294737505799*pi) q[25];
U1q(0.302580219287146*pi,0.5134140563542*pi) q[26];
U1q(0.552292683409449*pi,1.60401349662832*pi) q[27];
U1q(0.83980913167738*pi,1.5864136411499201*pi) q[28];
U1q(0.268124042150477*pi,0.3938516421414002*pi) q[29];
U1q(0.339977580115*pi,0.4444580169050001*pi) q[30];
U1q(0.686212004038847*pi,0.13472704134885038*pi) q[31];
U1q(0.220900319989771*pi,0.31611804419561995*pi) q[32];
U1q(0.158002528614776*pi,1.8367996283623702*pi) q[33];
U1q(0.594492304661002*pi,1.8026785722933196*pi) q[34];
U1q(0.706191019238726*pi,1.5977729997316104*pi) q[35];
U1q(0.439440528324023*pi,1.7479552107374197*pi) q[36];
U1q(0.19257136138611*pi,0.20898801122363997*pi) q[37];
U1q(0.749497327757347*pi,0.9543922837458396*pi) q[38];
U1q(0.827870301277818*pi,0.67550160179607*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[21],q[3];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[19],q[11];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[32],q[13];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[27],q[36];
RZZ(0.5*pi) q[35],q[33];
U1q(0.620267096290276*pi,0.06228595794668035*pi) q[0];
U1q(0.757730579981025*pi,0.5092729426056399*pi) q[1];
U1q(0.532330807795808*pi,0.21232633812523005*pi) q[2];
U1q(0.324124009039242*pi,1.18545716240049*pi) q[3];
U1q(0.735959298816915*pi,1.69899461212559*pi) q[4];
U1q(0.132845745587006*pi,0.9606798580569897*pi) q[5];
U1q(0.593405927177347*pi,0.9531014972818497*pi) q[6];
U1q(0.0800280740160646*pi,0.0681351025985002*pi) q[7];
U1q(0.75065216485709*pi,1.6361401388097798*pi) q[8];
U1q(0.54183556627118*pi,1.2475151912810896*pi) q[9];
U1q(0.298282078841708*pi,0.4492233682040201*pi) q[10];
U1q(0.86576298045049*pi,1.4378979822522897*pi) q[11];
U1q(0.370316191097358*pi,0.21923117697649008*pi) q[12];
U1q(0.765023256044338*pi,1.2013543701572509*pi) q[13];
U1q(0.315460269431359*pi,0.07062049915641033*pi) q[14];
U1q(0.20891779201409*pi,0.7106262738021103*pi) q[15];
U1q(0.401734751830489*pi,1.9069525927751005*pi) q[16];
U1q(0.676377777489941*pi,1.70005217438485*pi) q[17];
U1q(0.558570546478213*pi,0.3121975458045396*pi) q[18];
U1q(0.778649420020123*pi,0.23022175346445017*pi) q[19];
U1q(0.383306097763154*pi,1.7155640140409094*pi) q[20];
U1q(0.658526244914292*pi,0.8674394210069103*pi) q[21];
U1q(0.403954946625799*pi,0.3452715422985202*pi) q[22];
U1q(0.354271584179183*pi,0.0007723254907903154*pi) q[23];
U1q(0.818124996352253*pi,1.5287743468886204*pi) q[24];
U1q(0.616841399493773*pi,0.3520760879892899*pi) q[25];
U1q(0.765265183991934*pi,1.51122726035761*pi) q[26];
U1q(0.675842239353502*pi,0.6372479314581998*pi) q[27];
U1q(0.315458308365468*pi,0.1753281942272702*pi) q[28];
U1q(0.863247907003062*pi,1.9257029001305002*pi) q[29];
U1q(0.802516315848745*pi,0.12546366814617027*pi) q[30];
U1q(0.281667479787057*pi,1.2875888905691006*pi) q[31];
U1q(0.364320263826553*pi,0.21196739466756043*pi) q[32];
U1q(0.842588786480978*pi,0.21467205362464004*pi) q[33];
U1q(0.0854471161632129*pi,1.3896668420865002*pi) q[34];
U1q(0.706732038599503*pi,1.2691143759796493*pi) q[35];
U1q(0.634719737834905*pi,0.06555151641703993*pi) q[36];
U1q(0.816823689879443*pi,0.9482403904112502*pi) q[37];
U1q(0.890749166803553*pi,1.2164657393207197*pi) q[38];
U1q(0.110073936610296*pi,1.13497140183549*pi) q[39];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[29],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[30],q[39];
RZZ(0.5*pi) q[37],q[35];
U1q(0.224211173797594*pi,1.0413953380043992*pi) q[0];
U1q(0.977611277820704*pi,1.0947000089541206*pi) q[1];
U1q(0.0654674886326824*pi,0.6265577093220802*pi) q[2];
U1q(0.465525287476438*pi,1.7105020727644007*pi) q[3];
U1q(0.363582878038459*pi,0.3961241821210706*pi) q[4];
U1q(0.354556470137539*pi,1.9482383830351395*pi) q[5];
U1q(0.182666507243794*pi,0.7342952189291996*pi) q[6];
U1q(0.486988130772203*pi,0.12330375701557017*pi) q[7];
U1q(0.273177466865353*pi,1.6004251793610997*pi) q[8];
U1q(0.808945683455279*pi,1.3065507297622005*pi) q[9];
U1q(0.381756530680452*pi,0.8243324090735005*pi) q[10];
U1q(0.290588875557729*pi,1.9103505580875897*pi) q[11];
U1q(0.719035789249577*pi,1.7861627036872303*pi) q[12];
U1q(0.500067386076942*pi,1.4386730406025006*pi) q[13];
U1q(0.557104172173699*pi,1.3864441479387999*pi) q[14];
U1q(0.70554072351764*pi,0.0063549241484395225*pi) q[15];
U1q(0.461424488996816*pi,0.12796118653630018*pi) q[16];
U1q(0.403347141316508*pi,0.8143140389284707*pi) q[17];
U1q(0.261026346242499*pi,1.6909175630066002*pi) q[18];
U1q(0.738044303982363*pi,1.4242158093718498*pi) q[19];
U1q(0.207484163075552*pi,1.4992838556471995*pi) q[20];
U1q(0.57503296533921*pi,0.7156922603313003*pi) q[21];
U1q(0.543478967408*pi,1.7511597658691507*pi) q[22];
U1q(0.0708669188602045*pi,0.4653265975439904*pi) q[23];
U1q(0.830797064513557*pi,0.8160011632464004*pi) q[24];
U1q(0.522312443326932*pi,0.8726441631117297*pi) q[25];
U1q(0.266918810486264*pi,1.0089479603738898*pi) q[26];
U1q(0.356624663451241*pi,0.5811433882566899*pi) q[27];
U1q(0.589117456383601*pi,0.9792383693601598*pi) q[28];
U1q(0.904571147818341*pi,1.9859726155470092*pi) q[29];
U1q(0.82206373921712*pi,0.7486419406190308*pi) q[30];
U1q(0.203149328915948*pi,1.6535877000988002*pi) q[31];
U1q(0.638857243134321*pi,1.1170503589993697*pi) q[32];
U1q(0.753019244762279*pi,1.42327545032507*pi) q[33];
U1q(0.419610447655038*pi,0.7387218316866608*pi) q[34];
U1q(0.71690257572118*pi,0.3892287127926295*pi) q[35];
U1q(0.612029552757049*pi,1.4955334046586994*pi) q[36];
U1q(0.356114388376768*pi,1.7670765396694605*pi) q[37];
U1q(0.123848386394774*pi,0.07646447506080989*pi) q[38];
U1q(0.4456205435038*pi,1.6016208399291099*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[28],q[18];
RZZ(0.5*pi) q[19],q[38];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[31],q[36];
U1q(0.83247208650105*pi,0.9487479577508999*pi) q[0];
U1q(0.902469918952232*pi,0.17987437771603076*pi) q[1];
U1q(0.572696406395678*pi,0.27556263600263975*pi) q[2];
U1q(0.317275559135994*pi,1.8782209904446994*pi) q[3];
U1q(0.469339688015076*pi,1.1967029056515006*pi) q[4];
U1q(0.593831600908014*pi,1.2933038935204006*pi) q[5];
U1q(0.39955975806495*pi,1.0846110548369996*pi) q[6];
U1q(0.716171603339829*pi,0.8452668292104004*pi) q[7];
U1q(0.426304283132062*pi,0.8832507559299003*pi) q[8];
U1q(0.404921462291478*pi,0.7662175394955995*pi) q[9];
U1q(0.779322090016952*pi,1.2475662665366993*pi) q[10];
U1q(0.0734823835854781*pi,0.7752710418884003*pi) q[11];
U1q(0.547434467145769*pi,0.05405614506689993*pi) q[12];
U1q(0.292138472995529*pi,1.7993394193974996*pi) q[13];
U1q(0.824633373136355*pi,0.8672971453197995*pi) q[14];
U1q(0.985028202705809*pi,1.8380480117165998*pi) q[15];
U1q(0.867833787656884*pi,0.6879667760101*pi) q[16];
U1q(0.260535501469424*pi,0.9884461950854*pi) q[17];
U1q(0.849432807381257*pi,1.8848868048041005*pi) q[18];
U1q(0.274782173278292*pi,0.3238551198720101*pi) q[19];
U1q(0.727117750425336*pi,0.42263023593790017*pi) q[20];
U1q(0.355027944003806*pi,0.09299219013269955*pi) q[21];
U1q(0.644223625744472*pi,1.1907314601022296*pi) q[22];
U1q(0.399040013010072*pi,1.2790714483241992*pi) q[23];
U1q(0.293198475997824*pi,0.9931761608493304*pi) q[24];
U1q(0.528920086804262*pi,0.7879957247681197*pi) q[25];
U1q(0.276834544030621*pi,0.35732192629637005*pi) q[26];
U1q(0.140243714697492*pi,1.9486566217050996*pi) q[27];
U1q(0.798089507141316*pi,1.99848865405452*pi) q[28];
U1q(0.668765423012813*pi,1.1065943222575498*pi) q[29];
U1q(0.701613989215521*pi,0.9119694701543999*pi) q[30];
U1q(0.688856628780657*pi,0.12107489102180047*pi) q[31];
U1q(0.510211167627223*pi,0.7152321629864993*pi) q[32];
U1q(0.320905169023878*pi,1.4586013747153501*pi) q[33];
U1q(0.458408359407469*pi,0.40046076127389973*pi) q[34];
U1q(0.622749313971609*pi,0.5425791381580005*pi) q[35];
U1q(0.249500097014367*pi,0.988238656319*pi) q[36];
U1q(0.836519961977691*pi,0.7182465658521995*pi) q[37];
U1q(0.614111624482101*pi,1.2449146249044993*pi) q[38];
U1q(0.491625832521142*pi,0.9619993204975508*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[19];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[28],q[11];
RZZ(0.5*pi) q[12],q[37];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[17],q[18];
RZZ(0.5*pi) q[30],q[21];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[26],q[24];
RZZ(0.5*pi) q[31],q[32];
RZZ(0.5*pi) q[39],q[33];
RZZ(0.5*pi) q[34],q[38];
U1q(0.91202267705938*pi,1.4648425810719008*pi) q[0];
U1q(0.347094593121161*pi,0.4708529941395998*pi) q[1];
U1q(0.632754194357682*pi,0.3688839512001998*pi) q[2];
U1q(0.273985715326528*pi,0.6610647373535006*pi) q[3];
U1q(0.467256661952302*pi,0.6257446128264998*pi) q[4];
U1q(0.293271801666516*pi,1.3100163697613993*pi) q[5];
U1q(0.22220177525826*pi,0.18605395388160062*pi) q[6];
U1q(0.858616737144379*pi,0.06867364012899912*pi) q[7];
U1q(0.566024497940244*pi,0.3574200424472007*pi) q[8];
U1q(0.564898457981224*pi,1.2827112138452001*pi) q[9];
U1q(0.664542235611285*pi,1.7415252346264012*pi) q[10];
U1q(0.516029107656969*pi,0.11666805437839933*pi) q[11];
U1q(0.667314642766958*pi,1.3946096576644003*pi) q[12];
U1q(0.704452093156383*pi,0.9291087085880996*pi) q[13];
U1q(0.499051272424091*pi,0.8506676366947996*pi) q[14];
U1q(0.176428621023325*pi,0.026506769940500163*pi) q[15];
U1q(0.305901702797349*pi,0.16096909989619945*pi) q[16];
U1q(0.555147926930803*pi,1.1304008526415998*pi) q[17];
U1q(0.57247740824757*pi,0.07964690541120056*pi) q[18];
U1q(0.740103486121403*pi,0.3729659547613906*pi) q[19];
U1q(0.271727771845459*pi,1.9606326861042014*pi) q[20];
U1q(0.512095571989003*pi,0.19673177835050026*pi) q[21];
U1q(0.078802800973272*pi,1.6707590401681998*pi) q[22];
U1q(0.527787975535664*pi,0.612020393261*pi) q[23];
U1q(0.401979617837936*pi,1.6543118700306003*pi) q[24];
U1q(0.433497998959318*pi,1.9164947680931892*pi) q[25];
U1q(0.265757265379806*pi,0.17698820222535083*pi) q[26];
U1q(0.488950929862233*pi,1.7195363041451994*pi) q[27];
U1q(0.26657250435486*pi,1.0601957188308102*pi) q[28];
U1q(0.60636142521147*pi,1.3015559041966007*pi) q[29];
U1q(0.539364200172632*pi,0.003331423763400565*pi) q[30];
U1q(0.263134777319071*pi,0.2730700301598006*pi) q[31];
U1q(0.661468138592642*pi,0.7492929926462004*pi) q[32];
U1q(0.513586292409315*pi,0.4620287299083703*pi) q[33];
U1q(0.391688677154669*pi,1.760067552977599*pi) q[34];
U1q(0.257410191912534*pi,1.0560547725168998*pi) q[35];
U1q(0.460850680831851*pi,0.16998465171380062*pi) q[36];
U1q(0.596223661436928*pi,1.2324040852235*pi) q[37];
U1q(0.551877829393975*pi,1.4863869945167991*pi) q[38];
U1q(0.406926816029779*pi,0.30863490712039976*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[19],q[1];
RZZ(0.5*pi) q[27],q[2];
RZZ(0.5*pi) q[17],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[12],q[23];
RZZ(0.5*pi) q[15],q[13];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[29],q[20];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[34],q[37];
RZZ(0.5*pi) q[36],q[38];
U1q(0.307507790649532*pi,0.5308506139197*pi) q[0];
U1q(0.0802858829626128*pi,1.9340254407248008*pi) q[1];
U1q(0.410712173668179*pi,1.2715277131558*pi) q[2];
U1q(0.309560716913689*pi,0.9815618972878006*pi) q[3];
U1q(0.837613070309383*pi,1.3539075351227012*pi) q[4];
U1q(0.694612327915303*pi,0.897293773392601*pi) q[5];
U1q(0.757502865995067*pi,0.4538815186499008*pi) q[6];
U1q(0.704654876898866*pi,1.2674352147393009*pi) q[7];
U1q(0.737678510498295*pi,1.8323097936393005*pi) q[8];
U1q(0.0544631687893364*pi,0.20613904717020048*pi) q[9];
U1q(0.50653060254774*pi,1.6322834281956986*pi) q[10];
U1q(0.774875581894646*pi,0.06034090407639958*pi) q[11];
U1q(0.549773007416622*pi,0.8615572918848997*pi) q[12];
U1q(0.58724005288062*pi,1.8581829083979997*pi) q[13];
U1q(0.533171135750354*pi,1.6028566013359011*pi) q[14];
U1q(0.327595309363576*pi,0.10440110063870023*pi) q[15];
U1q(0.250026134357891*pi,0.7440156664764004*pi) q[16];
U1q(0.267375943891957*pi,1.9708294476716013*pi) q[17];
U1q(0.71724094393401*pi,0.7434636692597998*pi) q[18];
U1q(0.327890657306341*pi,1.0461654861819003*pi) q[19];
U1q(0.721419914633526*pi,1.1454643418711008*pi) q[20];
U1q(0.296276492384975*pi,0.35087817763050033*pi) q[21];
U1q(0.672941506544301*pi,0.055134342627200184*pi) q[22];
U1q(0.258094075564655*pi,1.645438901984999*pi) q[23];
U1q(0.494583847063814*pi,1.8791243066426002*pi) q[24];
U1q(0.662255611583187*pi,1.8937559135383992*pi) q[25];
U1q(0.674505197401304*pi,0.5754449540274997*pi) q[26];
U1q(0.90109821994763*pi,0.8247781638771006*pi) q[27];
U1q(0.37461092209923*pi,1.0089142928135004*pi) q[28];
U1q(0.432113021302259*pi,0.6813339230423008*pi) q[29];
U1q(0.375821195905066*pi,1.3636398121869*pi) q[30];
U1q(0.364119063270516*pi,1.5483533101402003*pi) q[31];
U1q(0.471246001513706*pi,0.9265488653590985*pi) q[32];
U1q(0.907897889415278*pi,0.6461909045905294*pi) q[33];
U1q(0.935188043780002*pi,1.9306881577762987*pi) q[34];
U1q(0.643977430239592*pi,1.358869414451*pi) q[35];
U1q(0.599442735658391*pi,0.6987437537947017*pi) q[36];
U1q(0.63040753626142*pi,0.7989548670330997*pi) q[37];
U1q(0.60340150137957*pi,1.7756287751670001*pi) q[38];
U1q(0.36443660617244*pi,1.8588193249322007*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[6],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[16],q[28];
RZZ(0.5*pi) q[31],q[18];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[37],q[24];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[32],q[33];
U1q(0.595558726081631*pi,1.1348620934907991*pi) q[0];
U1q(0.407528069565664*pi,0.021215998634000144*pi) q[1];
U1q(0.42730638684359*pi,0.8641319799905993*pi) q[2];
U1q(0.319064394440236*pi,0.32553612738849935*pi) q[3];
U1q(0.527933777848453*pi,0.467625561236499*pi) q[4];
U1q(0.269012681932619*pi,0.876951720738699*pi) q[5];
U1q(0.161578461850589*pi,0.1449623088011016*pi) q[6];
U1q(0.428479227628165*pi,1.0540598124085*pi) q[7];
U1q(0.173480677337*pi,0.5531547391952003*pi) q[8];
U1q(0.240860261394565*pi,0.3488923985017003*pi) q[9];
U1q(0.374879031264365*pi,0.5512095775749017*pi) q[10];
U1q(0.625812280238211*pi,1.6912972991357016*pi) q[11];
U1q(0.669504338579133*pi,1.1099307480454002*pi) q[12];
U1q(0.469911833673916*pi,0.11098147832489857*pi) q[13];
U1q(0.403542332389974*pi,0.4217841834547009*pi) q[14];
U1q(0.529882602017547*pi,1.0123671668421004*pi) q[15];
U1q(0.540604590562049*pi,0.5675859099479013*pi) q[16];
U1q(0.575659028906235*pi,1.1446488752360011*pi) q[17];
U1q(0.363653071853018*pi,0.4036026756451001*pi) q[18];
U1q(0.648539717156702*pi,1.5558891413518001*pi) q[19];
U1q(0.480212563533785*pi,1.8137922483635016*pi) q[20];
U1q(0.307711983263749*pi,1.7198720553115017*pi) q[21];
U1q(0.347651223585917*pi,0.3364056638551993*pi) q[22];
U1q(0.427494227571383*pi,0.07395008268689907*pi) q[23];
U1q(0.273708078340612*pi,0.4830807609173*pi) q[24];
U1q(0.148140244913189*pi,0.9568589004739998*pi) q[25];
U1q(0.27210841447648*pi,0.9577629723297996*pi) q[26];
U1q(0.416011813378661*pi,1.8978707227598015*pi) q[27];
U1q(0.710374723268001*pi,1.9387905496647004*pi) q[28];
U1q(0.767830671255543*pi,1.9511957805302007*pi) q[29];
U1q(0.133880573910395*pi,0.4493606401294006*pi) q[30];
U1q(0.331839591161118*pi,0.9293541819201003*pi) q[31];
U1q(0.653049764267758*pi,1.996639150920199*pi) q[32];
U1q(0.269751715393695*pi,1.0027625893418008*pi) q[33];
U1q(0.212399099533203*pi,1.4511917106711998*pi) q[34];
U1q(0.906383749995547*pi,0.8297033800563014*pi) q[35];
U1q(0.729708101673303*pi,0.29591514569489874*pi) q[36];
U1q(0.421110585657134*pi,0.34610762457370114*pi) q[37];
U1q(0.625961680111001*pi,0.33982437750729844*pi) q[38];
U1q(0.465015851690683*pi,0.8279910066525993*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[36],q[1];
RZZ(0.5*pi) q[2],q[33];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[9],q[15];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[21],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[20],q[32];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[30],q[29];
U1q(0.25437905492894*pi,1.3006962227501013*pi) q[0];
U1q(0.595996123677791*pi,0.9599266542121008*pi) q[1];
U1q(0.617510176971555*pi,1.0829190484546984*pi) q[2];
U1q(0.254311881136874*pi,0.07331629688350105*pi) q[3];
U1q(0.499161252012278*pi,0.02249255614970025*pi) q[4];
U1q(0.228587071022156*pi,0.711513479419299*pi) q[5];
U1q(0.694327699340243*pi,1.4234343229376982*pi) q[6];
U1q(0.674549888425389*pi,0.46426734097050115*pi) q[7];
U1q(0.0446765580238918*pi,0.7566304677357998*pi) q[8];
U1q(0.288957390656852*pi,0.4997065862896015*pi) q[9];
U1q(0.802273082855844*pi,0.7754758715873002*pi) q[10];
U1q(0.715706459284894*pi,0.19945551334179967*pi) q[11];
U1q(0.485182613389535*pi,0.034859044959500096*pi) q[12];
U1q(0.924726293308406*pi,1.763292876691601*pi) q[13];
U1q(0.866575317907863*pi,1.8909069733178008*pi) q[14];
U1q(0.919333651131423*pi,0.3970621970310013*pi) q[15];
U1q(0.629561737674101*pi,1.7171346510184016*pi) q[16];
U1q(0.811637102844592*pi,0.15749512218749828*pi) q[17];
U1q(0.669238882245068*pi,1.6190950047041*pi) q[18];
U1q(0.491191549721279*pi,0.9648120165583993*pi) q[19];
U1q(0.332929543392033*pi,1.287062542138301*pi) q[20];
U1q(0.475013434296608*pi,0.5907194163173983*pi) q[21];
U1q(0.610721026803758*pi,0.04422368483639971*pi) q[22];
U1q(0.693519638407266*pi,0.39049737735070167*pi) q[23];
U1q(0.561408305059478*pi,0.5767819870554014*pi) q[24];
U1q(0.407628819976091*pi,0.8966356676294005*pi) q[25];
U1q(0.340595686114267*pi,0.6287633846947003*pi) q[26];
U1q(0.663282402556957*pi,1.0231564343072996*pi) q[27];
U1q(0.725284059662003*pi,0.8353702847280999*pi) q[28];
U1q(0.525160521538427*pi,0.7041850048141001*pi) q[29];
U1q(0.444391608392525*pi,1.6303364644662999*pi) q[30];
U1q(0.670754667058617*pi,0.7875459291314009*pi) q[31];
U1q(0.399764488925616*pi,1.7043042788286016*pi) q[32];
U1q(0.604233785950726*pi,0.9788552962910995*pi) q[33];
U1q(0.627656681453886*pi,0.27680241680310047*pi) q[34];
U1q(0.529110418563284*pi,1.5190047613562*pi) q[35];
U1q(0.422003308763567*pi,1.9139567032196005*pi) q[36];
U1q(0.415098553532131*pi,0.853311928019501*pi) q[37];
U1q(0.300567639348621*pi,1.5376308856411*pi) q[38];
U1q(0.384967315636344*pi,0.16890237886590143*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[25],q[2];
RZZ(0.5*pi) q[11],q[3];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[16];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[27],q[35];
RZZ(0.5*pi) q[29],q[33];
U1q(0.156002759357537*pi,1.9418034669716988*pi) q[0];
U1q(0.368081743981502*pi,0.24847311073369838*pi) q[1];
U1q(0.344691364231101*pi,1.2299898571709988*pi) q[2];
U1q(0.652983520321003*pi,0.0796625531799009*pi) q[3];
U1q(0.269045564593606*pi,1.7996599278705006*pi) q[4];
U1q(0.274302532504247*pi,1.3512223974302984*pi) q[5];
U1q(0.906148849888213*pi,1.8991385458941004*pi) q[6];
U1q(0.370382874152235*pi,1.6875985564009994*pi) q[7];
U1q(0.46511395859745*pi,1.5769433592965*pi) q[8];
U1q(0.75985536672732*pi,1.1959918539087013*pi) q[9];
U1q(0.248569423467465*pi,0.7231689714234015*pi) q[10];
U1q(0.446416012303287*pi,0.7764740831365984*pi) q[11];
U1q(0.639981408387553*pi,1.9148805976135996*pi) q[12];
U1q(0.358817640639928*pi,0.08917613693079929*pi) q[13];
U1q(0.173124160538236*pi,1.795965772009101*pi) q[14];
U1q(0.407592063886633*pi,1.8574308668342*pi) q[15];
U1q(0.552150422507802*pi,1.0881821691835007*pi) q[16];
U1q(0.739139761702185*pi,1.1583681216510016*pi) q[17];
U1q(0.582456956099541*pi,0.17012803261760112*pi) q[18];
U1q(0.433656567768697*pi,0.054124026182698515*pi) q[19];
U1q(0.187252668998903*pi,0.5171459537591012*pi) q[20];
U1q(0.185880016609029*pi,1.0351091167120998*pi) q[21];
U1q(0.564536604579015*pi,0.7109086060023984*pi) q[22];
U1q(0.63722747071237*pi,0.9603379465286999*pi) q[23];
U1q(0.208045317596187*pi,1.9466428012972017*pi) q[24];
U1q(0.515735388233099*pi,1.1336484244869993*pi) q[25];
U1q(0.275361072927449*pi,0.29119355800339974*pi) q[26];
U1q(0.518953721638988*pi,0.8967233252292992*pi) q[27];
U1q(0.324297248417413*pi,0.6329043278587996*pi) q[28];
U1q(0.436970002397366*pi,0.6925867953762008*pi) q[29];
U1q(0.405487073829725*pi,1.0054215167567015*pi) q[30];
U1q(0.334928725253581*pi,0.4498964848512017*pi) q[31];
U1q(0.0920964749421084*pi,1.7907050104318998*pi) q[32];
U1q(0.475154226576395*pi,0.5663169328059006*pi) q[33];
U1q(0.472402617765666*pi,0.18274502557489924*pi) q[34];
U1q(0.67836604006406*pi,0.697644862576901*pi) q[35];
U1q(0.850218205975713*pi,1.518332816679301*pi) q[36];
U1q(0.543739457496822*pi,1.9048193902766002*pi) q[37];
U1q(0.942634518969373*pi,1.8842248325723006*pi) q[38];
U1q(0.696699514758662*pi,0.510842643533099*pi) q[39];
rz(3.7101819024969984*pi) q[0];
rz(2.308379626243301*pi) q[1];
rz(2.3062388841180983*pi) q[2];
rz(1.8305122761282995*pi) q[3];
rz(2.0795807972781013*pi) q[4];
rz(0.8960035204498986*pi) q[5];
rz(2.3394585928237*pi) q[6];
rz(1.5757389902463004*pi) q[7];
rz(0.7504387398222008*pi) q[8];
rz(0.060024140627501055*pi) q[9];
rz(3.8291731015513015*pi) q[10];
rz(2.0227455957590017*pi) q[11];
rz(3.7277704834807004*pi) q[12];
rz(1.3742236179255016*pi) q[13];
rz(2.4208626566966984*pi) q[14];
rz(3.7018479137088*pi) q[15];
rz(1.6432158044062*pi) q[16];
rz(0.6117760825401*pi) q[17];
rz(2.3504127750964017*pi) q[18];
rz(3.3874148616712*pi) q[19];
rz(0.0024389107788991282*pi) q[20];
rz(1.7192078411048008*pi) q[21];
rz(1.5355983777524003*pi) q[22];
rz(0.4313423005628998*pi) q[23];
rz(1.2229468826696*pi) q[24];
rz(2.1415740348251013*pi) q[25];
rz(3.3919917820554986*pi) q[26];
rz(3.2293097049218993*pi) q[27];
rz(0.6236403935055996*pi) q[28];
rz(1.0237616943579013*pi) q[29];
rz(0.47916603603379926*pi) q[30];
rz(1.9834465719572982*pi) q[31];
rz(1.738551562525501*pi) q[32];
rz(2.5596057857764*pi) q[33];
rz(1.1650525920257984*pi) q[34];
rz(0.36327616709059996*pi) q[35];
rz(0.5548979054069001*pi) q[36];
rz(3.3802421302968*pi) q[37];
rz(1.0906116736945002*pi) q[38];
rz(2.303284390092699*pi) q[39];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];