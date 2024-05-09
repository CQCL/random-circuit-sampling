OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
U1q(0.430240747700963*pi,1.034858733375813*pi) q[0];
U1q(0.726596500329759*pi,0.591520809215408*pi) q[1];
U1q(0.201546661269782*pi,0.66320922298599*pi) q[2];
U1q(0.185146651322242*pi,1.478417547669404*pi) q[3];
U1q(0.832875006503812*pi,0.550774527890715*pi) q[4];
U1q(0.338223659945875*pi,0.567339411867821*pi) q[5];
U1q(0.63008653184699*pi,1.875120147985677*pi) q[6];
U1q(0.733366527882713*pi,1.620946855561561*pi) q[7];
U1q(0.778089104604528*pi,1.347161855670399*pi) q[8];
U1q(0.49567979988782*pi,0.941471889256088*pi) q[9];
U1q(0.440637860233642*pi,0.351461559312327*pi) q[10];
U1q(0.368796692173326*pi,0.930082216913724*pi) q[11];
U1q(0.660944778219298*pi,1.612889386739229*pi) q[12];
U1q(0.64956965971898*pi,1.807498337088658*pi) q[13];
U1q(0.23714394839318*pi,1.588364956966378*pi) q[14];
U1q(0.539490789720991*pi,0.78666292967373*pi) q[15];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[11],q[1];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[14],q[15];
U1q(0.326191272396277*pi,0.12036748487758997*pi) q[0];
U1q(0.723530355643617*pi,0.6755441015157599*pi) q[1];
U1q(0.698254892034853*pi,1.27785910808656*pi) q[2];
U1q(0.775228026339891*pi,0.38556205214874995*pi) q[3];
U1q(0.610089658021817*pi,0.7174916933759199*pi) q[4];
U1q(0.776431519676185*pi,1.50843481627891*pi) q[5];
U1q(0.518026223495166*pi,1.44144191311615*pi) q[6];
U1q(0.732610864028372*pi,0.9949345911854501*pi) q[7];
U1q(0.38894651960375*pi,1.2992359138015397*pi) q[8];
U1q(0.228001283271887*pi,0.54177998362517*pi) q[9];
U1q(0.522541916525378*pi,0.9552721081492699*pi) q[10];
U1q(0.698138720243406*pi,1.248461878703718*pi) q[11];
U1q(0.520119293760011*pi,0.5279331726897198*pi) q[12];
U1q(0.278060027784676*pi,0.25152825874705*pi) q[13];
U1q(0.658475901349009*pi,0.9273675414827101*pi) q[14];
U1q(0.277627821146308*pi,0.34293105652369005*pi) q[15];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[7];
RZZ(0.5*pi) q[11],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[14];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[12];
RZZ(0.5*pi) q[15],q[13];
U1q(0.490807591790842*pi,1.98918278993097*pi) q[0];
U1q(0.843952518268226*pi,0.65606758944991*pi) q[1];
U1q(0.795623483405344*pi,1.01781338218515*pi) q[2];
U1q(0.735971948937489*pi,1.9666601600860902*pi) q[3];
U1q(0.171034843172638*pi,0.9417141018289898*pi) q[4];
U1q(0.383322151266908*pi,1.3513753888571296*pi) q[5];
U1q(0.862402743893984*pi,0.02126471934305041*pi) q[6];
U1q(0.377434902139398*pi,1.7127112843993597*pi) q[7];
U1q(0.255342234227793*pi,1.2656818694613303*pi) q[8];
U1q(0.447570547318924*pi,1.5514790911998197*pi) q[9];
U1q(0.704258602312224*pi,0.9612200111231699*pi) q[10];
U1q(0.495586900001556*pi,0.041540807788859935*pi) q[11];
U1q(0.244146974217394*pi,0.42204089766087005*pi) q[12];
U1q(0.253870876676768*pi,0.16626820124433994*pi) q[13];
U1q(0.742875019813866*pi,0.3948771971557199*pi) q[14];
U1q(0.818566784434609*pi,0.73130740205927*pi) q[15];
RZZ(0.5*pi) q[0],q[12];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[2],q[15];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[14];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[10],q[8];
U1q(0.349848553033914*pi,0.1722275762813199*pi) q[0];
U1q(0.548472295798144*pi,0.6765491299969604*pi) q[1];
U1q(0.878108475668039*pi,0.5049710374158698*pi) q[2];
U1q(0.556944109769877*pi,1.3948019108594192*pi) q[3];
U1q(0.724332831454426*pi,0.3003674293703904*pi) q[4];
U1q(0.486071519771839*pi,0.37661446654191977*pi) q[5];
U1q(0.829036433994657*pi,1.5866423912078407*pi) q[6];
U1q(0.318976868383941*pi,1.7079478973487507*pi) q[7];
U1q(0.91802464666584*pi,0.6412912050531601*pi) q[8];
U1q(0.476126722552393*pi,0.1638935291397301*pi) q[9];
U1q(0.623899180624238*pi,1.2487608425038799*pi) q[10];
U1q(0.763000847771543*pi,0.17209673316565022*pi) q[11];
U1q(0.908703492893723*pi,0.6473677753588891*pi) q[12];
U1q(0.540415888754839*pi,0.08990279180777971*pi) q[13];
U1q(0.128841656563053*pi,1.00118821069168*pi) q[14];
U1q(0.724415216321011*pi,0.3635332604608301*pi) q[15];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[13];
U1q(0.855712737877118*pi,0.7899257399441293*pi) q[0];
U1q(0.292931848358425*pi,1.1788491659231104*pi) q[1];
U1q(0.37298797420355*pi,1.37447547778112*pi) q[2];
U1q(0.939114780084634*pi,0.38868866711360006*pi) q[3];
U1q(0.830321147649595*pi,1.3749431951965798*pi) q[4];
U1q(0.804539604230109*pi,1.0129718384827004*pi) q[5];
U1q(0.836859999397578*pi,0.6734128516212508*pi) q[6];
U1q(0.506548959279697*pi,1.2010281120962993*pi) q[7];
U1q(0.760241973461371*pi,0.6992373620423997*pi) q[8];
U1q(0.731796180364456*pi,1.9981156745877993*pi) q[9];
U1q(0.606444019900474*pi,1.1353459627309306*pi) q[10];
U1q(0.706935558035232*pi,1.9253685503934603*pi) q[11];
U1q(0.270461500946502*pi,0.9766921410523999*pi) q[12];
U1q(0.756426327238806*pi,1.7499927965086002*pi) q[13];
U1q(0.727388715455068*pi,0.38814786368594056*pi) q[14];
U1q(0.684400141591258*pi,0.5796293117537701*pi) q[15];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[13];
RZZ(0.5*pi) q[4],q[3];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[15],q[12];
U1q(0.368678901292588*pi,1.1446827521297998*pi) q[0];
U1q(0.562890663440951*pi,0.08322578614368936*pi) q[1];
U1q(0.228694171572402*pi,0.19512324103901957*pi) q[2];
U1q(0.48793378464627*pi,1.3450109781541002*pi) q[3];
U1q(0.163061135940482*pi,0.6054213968244904*pi) q[4];
U1q(0.355896712902539*pi,0.8748163500918*pi) q[5];
U1q(0.81915590314658*pi,0.26829716949199955*pi) q[6];
U1q(0.266312869961498*pi,0.7136479218274996*pi) q[7];
U1q(0.695827605252165*pi,0.21091441783470088*pi) q[8];
U1q(0.729619740104797*pi,0.4345097829767006*pi) q[9];
U1q(0.546859118150094*pi,1.7792647677283995*pi) q[10];
U1q(0.676033820497136*pi,1.7635796616221704*pi) q[11];
U1q(0.505939715796859*pi,1.7379538953828*pi) q[12];
U1q(0.104521400202223*pi,1.9581921035654197*pi) q[13];
U1q(0.407504431617034*pi,0.007888444784400761*pi) q[14];
U1q(0.33005214906653*pi,1.1028374823187104*pi) q[15];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[15];
RZZ(0.5*pi) q[14],q[13];
U1q(0.328859851488985*pi,0.9145075172960002*pi) q[0];
U1q(0.371286084657573*pi,1.0630412803868996*pi) q[1];
U1q(0.880041662749955*pi,0.97385080103974*pi) q[2];
U1q(0.811773938995988*pi,1.7081340187769989*pi) q[3];
U1q(0.173985104748114*pi,1.7199562485171995*pi) q[4];
U1q(0.346308379539004*pi,1.3520369371765995*pi) q[5];
U1q(0.763356802143883*pi,0.5791066283168007*pi) q[6];
U1q(0.12893404866502*pi,0.8460513667979015*pi) q[7];
U1q(0.787357793213247*pi,0.32605152987810015*pi) q[8];
U1q(0.160321437966425*pi,1.7225589433728992*pi) q[9];
U1q(0.651498019364853*pi,0.3543702132144002*pi) q[10];
U1q(0.380551584216429*pi,1.0810314268836105*pi) q[11];
U1q(0.372374391304871*pi,1.4645872203935006*pi) q[12];
U1q(0.618863222636837*pi,1.122844349628*pi) q[13];
U1q(0.380442909834495*pi,0.4927917026982005*pi) q[14];
U1q(0.386956473207634*pi,1.4687503165669007*pi) q[15];
RZZ(0.5*pi) q[0],q[13];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[2],q[8];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[9],q[15];
RZZ(0.5*pi) q[10],q[12];
U1q(0.325219851617456*pi,1.0346230002432009*pi) q[0];
U1q(0.495349491942223*pi,1.6921878580536998*pi) q[1];
U1q(0.487567421688573*pi,1.2994072768010998*pi) q[2];
U1q(0.594112054701962*pi,0.9077122329266984*pi) q[3];
U1q(0.71857497904198*pi,0.22946499576599955*pi) q[4];
U1q(0.72617528084657*pi,0.9834415077156002*pi) q[5];
U1q(0.208052299144443*pi,0.8013750246004001*pi) q[6];
U1q(0.178763378507405*pi,1.957275973855701*pi) q[7];
U1q(0.759503559345558*pi,1.1126400962507006*pi) q[8];
U1q(0.812406711735132*pi,1.2928056116630984*pi) q[9];
U1q(0.191820361462428*pi,1.7579026397073*pi) q[10];
U1q(0.159405372072591*pi,0.5675246761379*pi) q[11];
U1q(0.158127160786169*pi,0.14534486891900045*pi) q[12];
U1q(0.210217970916709*pi,0.23691662932109914*pi) q[13];
U1q(0.892778152523182*pi,0.6371080826001005*pi) q[14];
U1q(0.319570284146704*pi,1.3550657346607*pi) q[15];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[15];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[11],q[13];
U1q(0.624488779038595*pi,0.9854937651594007*pi) q[0];
U1q(0.606212506326167*pi,0.9326589994059002*pi) q[1];
U1q(0.47541692910198*pi,0.3255402146128006*pi) q[2];
U1q(0.349311140940314*pi,0.9213707649158991*pi) q[3];
U1q(0.497890833577183*pi,1.1030297074263995*pi) q[4];
U1q(0.667258870211082*pi,0.6292958935352999*pi) q[5];
U1q(0.357143018039991*pi,0.351304977878101*pi) q[6];
U1q(0.525281619055033*pi,0.24026784354710173*pi) q[7];
U1q(0.18385637677933*pi,1.2939918441825995*pi) q[8];
U1q(0.810059324672436*pi,1.413519737883199*pi) q[9];
U1q(0.73165212191162*pi,0.5300287720910006*pi) q[10];
U1q(0.502195379278959*pi,1.7846246700039998*pi) q[11];
U1q(0.84746934941446*pi,0.7431269689097988*pi) q[12];
U1q(0.379718880458037*pi,0.21849646912970044*pi) q[13];
U1q(0.772219644290073*pi,0.9095678403444012*pi) q[14];
U1q(0.31061670616782*pi,0.8961075684286008*pi) q[15];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[3];
RZZ(0.5*pi) q[2],q[7];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[11],q[15];
U1q(0.450341686284374*pi,0.3627262404725009*pi) q[0];
U1q(0.710865479054875*pi,1.8934874367181003*pi) q[1];
U1q(0.806700288544158*pi,0.9023261578710002*pi) q[2];
U1q(0.238901267079825*pi,0.3581475678961006*pi) q[3];
U1q(0.289047341301672*pi,0.20168113827759981*pi) q[4];
U1q(0.613109675460858*pi,1.9132701215443007*pi) q[5];
U1q(0.547665386604047*pi,1.9477917756429015*pi) q[6];
U1q(0.339514682096563*pi,1.9795749696456006*pi) q[7];
U1q(0.847469787016932*pi,1.1028221891165018*pi) q[8];
U1q(0.354382289538856*pi,1.772918898159201*pi) q[9];
U1q(0.73623151855668*pi,0.2640813870259997*pi) q[10];
U1q(0.372630995266719*pi,1.5935871796777992*pi) q[11];
U1q(0.708867835158339*pi,1.8311907010435*pi) q[12];
U1q(0.428521446936429*pi,1.4405629583438007*pi) q[13];
U1q(0.213177864743332*pi,1.3138442710946983*pi) q[14];
U1q(0.35916173850955*pi,0.8847289279426995*pi) q[15];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[9],q[7];
RZZ(0.5*pi) q[12],q[13];
U1q(0.843207199037027*pi,0.5130313475203998*pi) q[0];
U1q(0.243197454357575*pi,1.2503113370563987*pi) q[1];
U1q(0.292133652215213*pi,1.5004790639567993*pi) q[2];
U1q(0.612487578766338*pi,0.11479220671110113*pi) q[3];
U1q(0.304821395921101*pi,0.48153048305610113*pi) q[4];
U1q(0.357308423915488*pi,0.8119923262514988*pi) q[5];
U1q(0.883989238737473*pi,0.022054382320501276*pi) q[6];
U1q(0.703603858269508*pi,0.5816905685156009*pi) q[7];
U1q(0.681558680290332*pi,0.44174727970160177*pi) q[8];
U1q(0.286633585021934*pi,1.1911537914188983*pi) q[9];
U1q(0.562407645262648*pi,1.2904762511082986*pi) q[10];
U1q(0.477593962603581*pi,1.1106421514875997*pi) q[11];
U1q(0.171828560390767*pi,1.6490011010449983*pi) q[12];
U1q(0.225350112014997*pi,1.8775651060130016*pi) q[13];
U1q(0.476684979784384*pi,0.38931982531860143*pi) q[14];
U1q(0.922072612387758*pi,1.0337068515616998*pi) q[15];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[4],q[2];
RZZ(0.5*pi) q[5],q[7];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[11],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[10],q[13];
U1q(0.779158769384945*pi,0.7542333267948003*pi) q[0];
U1q(0.101287777289069*pi,0.8611254757135001*pi) q[1];
U1q(0.767927243503306*pi,0.4640685640027016*pi) q[2];
U1q(0.220284352664192*pi,0.11371997971849979*pi) q[3];
U1q(0.660207767350826*pi,0.47231921268679855*pi) q[4];
U1q(0.641532952919869*pi,0.052500485232901184*pi) q[5];
U1q(0.398414691893643*pi,0.6614826488074002*pi) q[6];
U1q(0.443410762188054*pi,1.2474473646589992*pi) q[7];
U1q(0.184529777795253*pi,1.0277007173034995*pi) q[8];
U1q(0.414530894292837*pi,1.9196325450016012*pi) q[9];
U1q(0.274930895810479*pi,0.36404847918609917*pi) q[10];
U1q(0.725060290401947*pi,1.7245385174934995*pi) q[11];
U1q(0.7664755861974*pi,1.1542932567368993*pi) q[12];
U1q(0.865156496775086*pi,0.9115610745758005*pi) q[13];
U1q(0.624085947137132*pi,1.0003336345282001*pi) q[14];
U1q(0.500128437508616*pi,1.0695976501384017*pi) q[15];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[8],q[13];
RZZ(0.5*pi) q[9],q[12];
U1q(0.424758740668737*pi,1.0628193289402006*pi) q[0];
U1q(0.638467576369771*pi,1.0302491369197*pi) q[1];
U1q(0.652317419449892*pi,1.3305056154298*pi) q[2];
U1q(0.309012485016368*pi,0.32097838551359814*pi) q[3];
U1q(0.19110965981621*pi,0.7910783032563984*pi) q[4];
U1q(0.718638868967483*pi,0.17633974628839866*pi) q[5];
U1q(0.549999107297047*pi,1.711539750870699*pi) q[6];
U1q(0.727504718946366*pi,1.535231742501999*pi) q[7];
U1q(0.105939973754646*pi,0.2431832375531009*pi) q[8];
U1q(0.52070798249989*pi,1.951664867382501*pi) q[9];
U1q(0.406272028559959*pi,1.4373425305412013*pi) q[10];
U1q(0.0945306332574304*pi,0.8374731950645007*pi) q[11];
U1q(0.654850570944325*pi,0.09090345063539829*pi) q[12];
U1q(0.93135327602902*pi,0.25004419600870165*pi) q[13];
U1q(0.436472001248769*pi,0.1536799186553992*pi) q[14];
U1q(0.416970197046566*pi,1.6806348336059003*pi) q[15];
rz(3.584757665407899*pi) q[0];
rz(0.2830930124832989*pi) q[1];
rz(3.7931163788846014*pi) q[2];
rz(2.6263888123027996*pi) q[3];
rz(2.2223665145682006*pi) q[4];
rz(3.2552045192536987*pi) q[5];
rz(2.3841799556618994*pi) q[6];
rz(1.1522795080868988*pi) q[7];
rz(3.715393115304*pi) q[8];
rz(1.7009465432195015*pi) q[9];
rz(3.1014940757058014*pi) q[10];
rz(3.6348470519406*pi) q[11];
rz(0.24352897082560077*pi) q[12];
rz(1.7701568625329998*pi) q[13];
rz(2.8471671984354003*pi) q[14];
rz(1.1378315685854012*pi) q[15];
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