OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.396708933035551*pi,0.59048110075714*pi) q[0];
U1q(0.876054059831751*pi,0.6238933821339*pi) q[1];
U1q(0.583829091316669*pi,0.9861478442223199*pi) q[2];
U1q(0.127928382442526*pi,0.08782567605803*pi) q[3];
U1q(0.338288107984596*pi,1.396650904468093*pi) q[4];
U1q(0.289217339625527*pi,1.073222578116795*pi) q[5];
U1q(0.375556438239205*pi,0.654617773365351*pi) q[6];
U1q(0.616686932830566*pi,0.946933633018789*pi) q[7];
U1q(0.897451396323694*pi,1.25933800766221*pi) q[8];
U1q(0.423777691490573*pi,1.3131023441495429*pi) q[9];
U1q(0.459095723832264*pi,0.313655902426866*pi) q[10];
U1q(0.269066724431378*pi,0.686102111650231*pi) q[11];
U1q(0.057930104667926*pi,1.4885757157038642*pi) q[12];
U1q(0.111985971313536*pi,1.809464396735538*pi) q[13];
U1q(0.424978305322871*pi,1.418101430940304*pi) q[14];
U1q(0.509743461175725*pi,1.758388283147731*pi) q[15];
U1q(0.753510218284393*pi,0.538693118964549*pi) q[16];
U1q(0.900956334258208*pi,1.3865891640558*pi) q[17];
U1q(0.233861816316436*pi,1.341544931970091*pi) q[18];
U1q(0.266393495732626*pi,0.747247086533644*pi) q[19];
U1q(0.468585084005516*pi,0.325049281547146*pi) q[20];
U1q(0.473817927813311*pi,0.512048155090879*pi) q[21];
U1q(0.150136991573921*pi,1.712520882345864*pi) q[22];
U1q(0.225393030428838*pi,0.6643265061219601*pi) q[23];
U1q(0.495837762056746*pi,1.908170943088076*pi) q[24];
U1q(0.565785754638864*pi,0.770374819491605*pi) q[25];
U1q(0.712558479311599*pi,1.318282069006477*pi) q[26];
U1q(0.860050328298634*pi,0.0420784887462873*pi) q[27];
U1q(0.850242052246779*pi,0.215459838692812*pi) q[28];
U1q(0.554125662144432*pi,1.11440243714363*pi) q[29];
U1q(0.404998591458762*pi,0.941211966455867*pi) q[30];
U1q(0.784511740731553*pi,1.632480128018051*pi) q[31];
U1q(0.844642590680708*pi,0.986901305680317*pi) q[32];
U1q(0.551762166590341*pi,1.0160959298068*pi) q[33];
U1q(0.449430135081967*pi,1.431698527791005*pi) q[34];
U1q(0.263830832855551*pi,0.85864566287598*pi) q[35];
U1q(0.678197658279406*pi,0.0933185046602817*pi) q[36];
U1q(0.0658232517428754*pi,0.419596129732156*pi) q[37];
U1q(0.694708117806441*pi,0.760127723297263*pi) q[38];
U1q(0.46742966891853*pi,0.4324101431082401*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[4],q[10];
RZZ(0.5*pi) q[12],q[5];
RZZ(0.5*pi) q[8],q[6];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[9],q[26];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[28],q[33];
RZZ(0.5*pi) q[30],q[32];
RZZ(0.5*pi) q[31],q[34];
RZZ(0.5*pi) q[35],q[37];
U1q(0.0960597249538655*pi,1.13559386141046*pi) q[0];
U1q(0.969185310989793*pi,0.47388290089655993*pi) q[1];
U1q(0.599744236595151*pi,1.7983675921593596*pi) q[2];
U1q(0.410725234925459*pi,1.28222647175014*pi) q[3];
U1q(0.331264589515466*pi,0.8452082021265499*pi) q[4];
U1q(0.534889831970019*pi,0.6010727819069801*pi) q[5];
U1q(0.794116364310297*pi,1.70847048489768*pi) q[6];
U1q(0.640320405670532*pi,1.858204563505122*pi) q[7];
U1q(0.783885344386029*pi,0.43519803553861003*pi) q[8];
U1q(0.374749698118221*pi,1.2499799614309102*pi) q[9];
U1q(0.584782491108546*pi,1.4008054926031002*pi) q[10];
U1q(0.813833652689523*pi,0.16082832201107*pi) q[11];
U1q(0.167629757509859*pi,1.2647232619704298*pi) q[12];
U1q(0.461875581463302*pi,1.1894133046938902*pi) q[13];
U1q(0.869444695975948*pi,0.8126212966581201*pi) q[14];
U1q(0.575420359492687*pi,1.154381931644058*pi) q[15];
U1q(0.456886323443174*pi,0.7964396161702001*pi) q[16];
U1q(0.689726475235564*pi,1.9048944648008077*pi) q[17];
U1q(0.712588270901784*pi,1.15699714925133*pi) q[18];
U1q(0.455222340283768*pi,1.5094167548483899*pi) q[19];
U1q(0.80294530885729*pi,1.2619329532502999*pi) q[20];
U1q(0.386812928218452*pi,1.71934401144307*pi) q[21];
U1q(0.425962741086168*pi,0.62905787971409*pi) q[22];
U1q(0.677706301294561*pi,0.9949745467845799*pi) q[23];
U1q(0.440775242242989*pi,1.7772720953585002*pi) q[24];
U1q(0.587509085813412*pi,0.520667206679118*pi) q[25];
U1q(0.806602191908855*pi,1.9505914163252296*pi) q[26];
U1q(0.594871122658515*pi,1.07342525969037*pi) q[27];
U1q(0.517586472755995*pi,1.175910183535918*pi) q[28];
U1q(0.866387002205268*pi,0.7806705231214699*pi) q[29];
U1q(0.737948790116084*pi,1.8023876762001398*pi) q[30];
U1q(0.853814417840022*pi,1.9887044068346604*pi) q[31];
U1q(0.37125918386259*pi,1.488577961138661*pi) q[32];
U1q(0.375877908808085*pi,0.09338813838335991*pi) q[33];
U1q(0.660288423170931*pi,1.76277485921476*pi) q[34];
U1q(0.708353984545618*pi,0.03937246960163998*pi) q[35];
U1q(0.336761319958954*pi,1.64264757701925*pi) q[36];
U1q(0.385665320434116*pi,1.1814095455550602*pi) q[37];
U1q(0.236651050236397*pi,1.804665320014432*pi) q[38];
U1q(0.264307782102093*pi,0.83513408706661*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[3],q[7];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[8],q[33];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[11],q[20];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[16],q[27];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[23],q[28];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[31],q[36];
U1q(0.707993308266964*pi,0.8111132715012799*pi) q[0];
U1q(0.24022371890378*pi,0.7547554829539198*pi) q[1];
U1q(0.266367378772431*pi,0.7757181653507699*pi) q[2];
U1q(0.268376381660289*pi,0.08539129446962956*pi) q[3];
U1q(0.150673068325494*pi,1.0447779783868896*pi) q[4];
U1q(0.690698974084219*pi,0.24753582123038997*pi) q[5];
U1q(0.231800383640552*pi,0.11838580724071956*pi) q[6];
U1q(0.206328455183395*pi,1.8822420091805698*pi) q[7];
U1q(0.396333163143355*pi,1.07349988362777*pi) q[8];
U1q(0.554600664458391*pi,1.8887114251057904*pi) q[9];
U1q(0.358966621555309*pi,1.6088751096478298*pi) q[10];
U1q(0.294203101855161*pi,1.6048376660302202*pi) q[11];
U1q(0.720587665735277*pi,0.4168283959654504*pi) q[12];
U1q(0.60819790094292*pi,1.6613604058844897*pi) q[13];
U1q(0.269933609148933*pi,0.90584585770207*pi) q[14];
U1q(0.613715609312403*pi,1.5845173540593*pi) q[15];
U1q(0.559199147951305*pi,1.0929069798655102*pi) q[16];
U1q(0.429668465270956*pi,1.0059070289849*pi) q[17];
U1q(0.437949281446122*pi,1.6402727037302096*pi) q[18];
U1q(0.72724718589055*pi,0.40488999512276*pi) q[19];
U1q(0.33700938123531*pi,1.13903892545608*pi) q[20];
U1q(0.864215190781169*pi,0.37001235474616*pi) q[21];
U1q(0.716513544415018*pi,1.0096458373467296*pi) q[22];
U1q(0.677852483713613*pi,0.03139465091484972*pi) q[23];
U1q(0.221360593541603*pi,0.8459725338831898*pi) q[24];
U1q(0.693654539454587*pi,1.884785271991903*pi) q[25];
U1q(0.25750640295607*pi,1.1846681144172697*pi) q[26];
U1q(0.981288971494887*pi,0.5756376258922402*pi) q[27];
U1q(0.927958846398056*pi,1.03223483476035*pi) q[28];
U1q(0.541860039877912*pi,0.19270988749376983*pi) q[29];
U1q(0.589959619989242*pi,1.4099065880471597*pi) q[30];
U1q(0.775578720528436*pi,1.7059253960735496*pi) q[31];
U1q(0.399440632602019*pi,0.7874107771571199*pi) q[32];
U1q(0.236700866741022*pi,1.66130216854057*pi) q[33];
U1q(0.696899028685622*pi,1.3607586963004201*pi) q[34];
U1q(0.567715754050615*pi,0.08689881352353002*pi) q[35];
U1q(0.526523477576942*pi,0.37113535107410023*pi) q[36];
U1q(0.764970345408422*pi,1.6819743149303603*pi) q[37];
U1q(0.352718660113015*pi,1.14215990407192*pi) q[38];
U1q(0.440592768559133*pi,1.6899704172942096*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[33];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[9],q[34];
RZZ(0.5*pi) q[22],q[10];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[38],q[13];
RZZ(0.5*pi) q[29],q[14];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[32],q[39];
U1q(0.704743302619978*pi,0.3588817783814999*pi) q[0];
U1q(0.341923165955051*pi,1.5812632151083594*pi) q[1];
U1q(0.398772494655616*pi,0.5988507899043096*pi) q[2];
U1q(0.512460504281352*pi,1.5257360707910994*pi) q[3];
U1q(0.0170056911390086*pi,0.6124121898437496*pi) q[4];
U1q(0.310663778595563*pi,1.0348649680979998*pi) q[5];
U1q(0.393038171358177*pi,0.24091762049081922*pi) q[6];
U1q(0.604936215934526*pi,1.2560427681711896*pi) q[7];
U1q(0.358851219581672*pi,1.5710137675532607*pi) q[8];
U1q(0.357597723588308*pi,1.4484989293618797*pi) q[9];
U1q(0.307498818071137*pi,0.05737515347907962*pi) q[10];
U1q(0.701919314866302*pi,1.36316243778725*pi) q[11];
U1q(0.205379197493374*pi,0.9491013584836203*pi) q[12];
U1q(0.811084640470931*pi,0.5940671403527196*pi) q[13];
U1q(0.948935780495256*pi,0.7750205356676396*pi) q[14];
U1q(0.562851675072044*pi,1.9832862245212404*pi) q[15];
U1q(0.220436728967639*pi,1.0435454730639897*pi) q[16];
U1q(0.23124459418355*pi,0.6533833423148501*pi) q[17];
U1q(0.825495528608839*pi,1.4287599038424696*pi) q[18];
U1q(0.844316135438371*pi,0.43981242063255976*pi) q[19];
U1q(0.544635494209358*pi,0.8467915688952203*pi) q[20];
U1q(0.126489260835134*pi,1.2028165095453103*pi) q[21];
U1q(0.529460813857374*pi,0.5914284077101*pi) q[22];
U1q(0.342240206922203*pi,0.7900192504540202*pi) q[23];
U1q(0.890329554696598*pi,1.67069947311938*pi) q[24];
U1q(0.603002905498567*pi,1.761795682032634*pi) q[25];
U1q(0.366721407312042*pi,1.5512436909005896*pi) q[26];
U1q(0.638714711418387*pi,1.2501017153250702*pi) q[27];
U1q(0.286120833392841*pi,0.5100852221303196*pi) q[28];
U1q(0.294663094934346*pi,1.0401476472618496*pi) q[29];
U1q(0.613821503429187*pi,1.3332341822370903*pi) q[30];
U1q(0.735805862259216*pi,1.3694541168207*pi) q[31];
U1q(0.363150343091848*pi,1.23081721083759*pi) q[32];
U1q(0.0428568700286122*pi,0.6021990920110598*pi) q[33];
U1q(0.446030763880525*pi,1.0454439790966*pi) q[34];
U1q(0.189150660557528*pi,1.7705778381504498*pi) q[35];
U1q(0.233414517346742*pi,0.44046507897856024*pi) q[36];
U1q(0.437405575842237*pi,0.4908498603148601*pi) q[37];
U1q(0.0802553035087029*pi,0.15287121740451015*pi) q[38];
U1q(0.50526054623754*pi,0.37032762477748005*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[1],q[5];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[28],q[10];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[34];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[16],q[35];
RZZ(0.5*pi) q[17],q[32];
RZZ(0.5*pi) q[22],q[21];
RZZ(0.5*pi) q[23],q[33];
RZZ(0.5*pi) q[30],q[24];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[38],q[37];
U1q(0.244665478385219*pi,0.44935570677816994*pi) q[0];
U1q(0.4192323173981*pi,0.6675461311765005*pi) q[1];
U1q(0.353698687409085*pi,0.11875240441210089*pi) q[2];
U1q(0.31062644869253*pi,0.08219309231099992*pi) q[3];
U1q(0.592951792786592*pi,0.5761473606047298*pi) q[4];
U1q(0.500318112783786*pi,1.1730622281659997*pi) q[5];
U1q(0.290720543187721*pi,1.2122948874387003*pi) q[6];
U1q(0.645323471913271*pi,0.05714258340107037*pi) q[7];
U1q(0.694426886582938*pi,0.6613639723790996*pi) q[8];
U1q(0.627056184281773*pi,0.23266675680809001*pi) q[9];
U1q(0.398396962447233*pi,0.21190049058390947*pi) q[10];
U1q(0.407747056929522*pi,1.4068641087608604*pi) q[11];
U1q(0.229615800433778*pi,0.6812262703907006*pi) q[12];
U1q(0.300831728443023*pi,0.5860804550564005*pi) q[13];
U1q(0.689440082823911*pi,1.9873605307550992*pi) q[14];
U1q(0.68309952476314*pi,1.5664552607264897*pi) q[15];
U1q(0.379699937072699*pi,0.055091082179809625*pi) q[16];
U1q(0.453606788100158*pi,1.49351872195302*pi) q[17];
U1q(0.716694718038034*pi,1.5357040026373294*pi) q[18];
U1q(0.548516733995739*pi,0.8519604580836297*pi) q[19];
U1q(0.423410300197356*pi,1.3057597320360994*pi) q[20];
U1q(0.621612897228965*pi,0.3058752540052003*pi) q[21];
U1q(0.515242076938639*pi,0.7200010860050696*pi) q[22];
U1q(0.510496974777711*pi,0.5110165256816792*pi) q[23];
U1q(0.449227386257452*pi,0.20810551197626026*pi) q[24];
U1q(0.721988244988597*pi,0.22857672645195004*pi) q[25];
U1q(0.251238714825451*pi,0.06813225567189996*pi) q[26];
U1q(0.276437656706305*pi,0.07465074205484967*pi) q[27];
U1q(0.930062634658553*pi,0.024985417833610413*pi) q[28];
U1q(0.282477681063332*pi,1.2900468715396496*pi) q[29];
U1q(0.0680219735178538*pi,1.1531472434209906*pi) q[30];
U1q(0.0997559293559848*pi,1.9984038325920004*pi) q[31];
U1q(0.142268797063471*pi,1.4548366150711*pi) q[32];
U1q(0.484683715434527*pi,0.7509489552108892*pi) q[33];
U1q(0.690839089804766*pi,0.23284955816528985*pi) q[34];
U1q(0.952763371211143*pi,0.10887873246658941*pi) q[35];
U1q(0.589918440347958*pi,0.3347166482632993*pi) q[36];
U1q(0.910024325306332*pi,0.4021691369109499*pi) q[37];
U1q(0.139964523671299*pi,0.2957835707864298*pi) q[38];
U1q(0.29598847531225*pi,0.7886628636856896*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[39],q[6];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[37];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[23],q[26];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[30],q[33];
RZZ(0.5*pi) q[38],q[31];
U1q(0.702998005245013*pi,0.03808842082061048*pi) q[0];
U1q(0.455529791321397*pi,0.36843810080920036*pi) q[1];
U1q(0.462298051966462*pi,0.05691208952270088*pi) q[2];
U1q(0.844785838141595*pi,0.44800192689269913*pi) q[3];
U1q(0.439654970644796*pi,1.1371987056336597*pi) q[4];
U1q(0.771335814364827*pi,0.2977208508194593*pi) q[5];
U1q(0.643248287435262*pi,1.7702491383977996*pi) q[6];
U1q(0.407645742445862*pi,1.9638281293358695*pi) q[7];
U1q(0.701930384671702*pi,1.9296137039216*pi) q[8];
U1q(0.47369703158254*pi,1.6259681393442005*pi) q[9];
U1q(0.436005968702916*pi,1.7592642138361008*pi) q[10];
U1q(0.574254693557528*pi,0.4583210987137907*pi) q[11];
U1q(0.471195855838804*pi,1.0268193163050992*pi) q[12];
U1q(0.575184232393492*pi,1.4703849276684*pi) q[13];
U1q(0.657023826903563*pi,1.7249819638656803*pi) q[14];
U1q(0.617580417580106*pi,1.1217693943971696*pi) q[15];
U1q(0.701984263261636*pi,0.7331475976170498*pi) q[16];
U1q(0.390485965242868*pi,0.044812517116950445*pi) q[17];
U1q(0.229704860616118*pi,1.3426074523629996*pi) q[18];
U1q(0.277639788826369*pi,0.2840287490221396*pi) q[19];
U1q(0.646338074828465*pi,1.1485678574772002*pi) q[20];
U1q(0.583408496181831*pi,1.0668720351526808*pi) q[21];
U1q(0.44339072137356*pi,1.4466570855152003*pi) q[22];
U1q(0.491515858355014*pi,0.14142167506940062*pi) q[23];
U1q(0.646024306062436*pi,0.036880776924380854*pi) q[24];
U1q(0.173064869752697*pi,0.6825303675271099*pi) q[25];
U1q(0.288281984169766*pi,1.5598868109606006*pi) q[26];
U1q(0.746320133224102*pi,0.3609210593356895*pi) q[27];
U1q(0.521710273126763*pi,1.5822385194223294*pi) q[28];
U1q(0.561548154224587*pi,0.7514556123707994*pi) q[29];
U1q(0.278109009907767*pi,0.5422451012987004*pi) q[30];
U1q(0.217448803650135*pi,0.7426743629171*pi) q[31];
U1q(0.56563220656457*pi,1.7085849153944004*pi) q[32];
U1q(0.255469669616391*pi,0.8929513928978992*pi) q[33];
U1q(0.797539940847411*pi,1.0674208270336498*pi) q[34];
U1q(0.628371063525212*pi,1.6137038662254994*pi) q[35];
U1q(0.303092921452599*pi,0.17569970879129926*pi) q[36];
U1q(0.624998156862547*pi,0.2177390975144906*pi) q[37];
U1q(0.0836698424074264*pi,0.13996510777315052*pi) q[38];
U1q(0.569665713955872*pi,1.6441936940319692*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[37];
RZZ(0.5*pi) q[29],q[4];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[6],q[10];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[23],q[11];
RZZ(0.5*pi) q[38],q[12];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[21],q[33];
RZZ(0.5*pi) q[27],q[39];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[35],q[32];
U1q(0.483041811617704*pi,1.665366339298*pi) q[0];
U1q(0.399883321594933*pi,1.5929963526913014*pi) q[1];
U1q(0.808317763421884*pi,0.6199985112379007*pi) q[2];
U1q(0.0473271381177642*pi,0.12089907670499933*pi) q[3];
U1q(0.516696833802878*pi,0.4646196108219005*pi) q[4];
U1q(0.725649396474305*pi,0.7845898583464006*pi) q[5];
U1q(0.178842909854924*pi,0.7383650857908997*pi) q[6];
U1q(0.333771735670044*pi,1.5650300612908996*pi) q[7];
U1q(0.534683774462077*pi,1.2553863155141016*pi) q[8];
U1q(0.411440176539483*pi,1.4972215058306997*pi) q[9];
U1q(0.438620099099879*pi,0.41577896355189914*pi) q[10];
U1q(0.845523149955384*pi,0.5591250927431997*pi) q[11];
U1q(0.537727993997189*pi,0.8303042614277008*pi) q[12];
U1q(0.67437048658625*pi,1.2755435255263006*pi) q[13];
U1q(0.59559713369552*pi,1.4896568713609994*pi) q[14];
U1q(0.641963227607549*pi,0.45938468271335964*pi) q[15];
U1q(0.425527262191305*pi,1.5691602641106996*pi) q[16];
U1q(0.404274487629463*pi,1.6135602636264004*pi) q[17];
U1q(0.623574958485516*pi,0.2636531222032996*pi) q[18];
U1q(0.422397192396274*pi,0.31245032342289925*pi) q[19];
U1q(0.417869544587034*pi,0.8147238574844984*pi) q[20];
U1q(0.815563377305975*pi,1.6505507015694008*pi) q[21];
U1q(0.494524078125326*pi,1.1375255820813006*pi) q[22];
U1q(0.665999985249304*pi,0.25615826225859983*pi) q[23];
U1q(0.403144316761685*pi,0.8298344208842003*pi) q[24];
U1q(0.543788265004032*pi,1.4606852828705001*pi) q[25];
U1q(0.647435297756581*pi,1.6657972670788013*pi) q[26];
U1q(0.632464396199737*pi,0.15486638231440075*pi) q[27];
U1q(0.879686712368269*pi,0.7947254068390102*pi) q[28];
U1q(0.129144311190741*pi,0.18969890310630078*pi) q[29];
U1q(0.616207777332236*pi,1.1606050313805003*pi) q[30];
U1q(0.179854935071734*pi,1.9210621252990983*pi) q[31];
U1q(0.887760571753053*pi,1.2725089692194*pi) q[32];
U1q(0.519714551949405*pi,1.7132500240824005*pi) q[33];
U1q(0.754039062832425*pi,0.29692512790104963*pi) q[34];
U1q(0.717180613022509*pi,0.1395098357183997*pi) q[35];
U1q(0.243051325086336*pi,0.7243239556260992*pi) q[36];
U1q(0.469955685091285*pi,0.4843945657106996*pi) q[37];
U1q(0.301596036209367*pi,0.14240646658210032*pi) q[38];
U1q(0.620448167804638*pi,0.6446437219412005*pi) q[39];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[27],q[1];
RZZ(0.5*pi) q[3],q[28];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[9],q[33];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[21],q[24];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[39],q[36];
U1q(0.626455631877756*pi,1.2256643951903001*pi) q[0];
U1q(0.758323906174197*pi,0.1076191483810014*pi) q[1];
U1q(0.36393184316333*pi,1.705266423711599*pi) q[2];
U1q(0.533436146954959*pi,1.7749489271485999*pi) q[3];
U1q(0.823934983002838*pi,0.2560563753613998*pi) q[4];
U1q(0.557625286345599*pi,0.8045151874901002*pi) q[5];
U1q(0.210438858638178*pi,0.661230498691701*pi) q[6];
U1q(0.385526754870335*pi,1.7887574142561995*pi) q[7];
U1q(0.387897189990361*pi,0.6822587931336983*pi) q[8];
U1q(0.791712931060024*pi,0.2869062248418004*pi) q[9];
U1q(0.684432905869496*pi,0.1914746925566*pi) q[10];
U1q(0.557493134203473*pi,1.1384182520145991*pi) q[11];
U1q(0.866450889038953*pi,1.8507361299170988*pi) q[12];
U1q(0.536354721029432*pi,0.25899835870220045*pi) q[13];
U1q(0.831444810096751*pi,0.6624413148174995*pi) q[14];
U1q(0.480332744263855*pi,0.7376301439068005*pi) q[15];
U1q(0.724155042022104*pi,0.4821347938678002*pi) q[16];
U1q(0.624243660548746*pi,0.8231581830452992*pi) q[17];
U1q(0.607000611770304*pi,0.46009870578919987*pi) q[18];
U1q(0.175958975333485*pi,1.7801115175970992*pi) q[19];
U1q(0.653179660233212*pi,0.6220769879470005*pi) q[20];
U1q(0.906238660125761*pi,0.7455245136840993*pi) q[21];
U1q(0.783887364758831*pi,1.2883471085762004*pi) q[22];
U1q(0.509666004745698*pi,1.0604170254546013*pi) q[23];
U1q(0.470492036564731*pi,1.7721247711205006*pi) q[24];
U1q(0.581476266157416*pi,0.3462075486518197*pi) q[25];
U1q(0.712691060835884*pi,1.539085448825201*pi) q[26];
U1q(0.679152167983826*pi,0.5842760620690992*pi) q[27];
U1q(0.863659492528451*pi,0.19813399089090034*pi) q[28];
U1q(0.152409203453139*pi,0.1279484335654999*pi) q[29];
U1q(0.483087829130529*pi,1.8882728223263996*pi) q[30];
U1q(0.248842827320124*pi,1.7575469318773003*pi) q[31];
U1q(0.30973367718674*pi,0.9668685481142987*pi) q[32];
U1q(0.488768642711567*pi,1.9875872278252*pi) q[33];
U1q(0.588289271990125*pi,0.24225697693949932*pi) q[34];
U1q(0.584055854260718*pi,1.0264794167263993*pi) q[35];
U1q(0.116614972792418*pi,0.7312181224564007*pi) q[36];
U1q(0.665430801755951*pi,1.7758441846305004*pi) q[37];
U1q(0.729891736745728*pi,0.6093657074251002*pi) q[38];
U1q(0.297789173468242*pi,0.8052650326873998*pi) q[39];
RZZ(0.5*pi) q[0],q[12];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[3],q[31];
RZZ(0.5*pi) q[37],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[7],q[6];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[35],q[26];
RZZ(0.5*pi) q[30],q[39];
RZZ(0.5*pi) q[34],q[36];
U1q(0.300034235391665*pi,1.2327995198048995*pi) q[0];
U1q(0.728051762502585*pi,1.2075002368244014*pi) q[1];
U1q(0.236300509230072*pi,0.017467562759499344*pi) q[2];
U1q(0.431992683966828*pi,1.9075376861450017*pi) q[3];
U1q(0.266150649858312*pi,0.9351182162636995*pi) q[4];
U1q(0.869188555887461*pi,0.15710512404920074*pi) q[5];
U1q(0.530261639756582*pi,1.2272460229958995*pi) q[6];
U1q(0.49179727228337*pi,1.5270584452464*pi) q[7];
U1q(0.526008554960465*pi,0.06227458872900016*pi) q[8];
U1q(0.568201326666706*pi,1.9015110761350016*pi) q[9];
U1q(0.394411393933094*pi,1.5915088515215992*pi) q[10];
U1q(0.654590015920595*pi,0.03046632908399971*pi) q[11];
U1q(0.638302988979017*pi,0.9760295817857987*pi) q[12];
U1q(0.35172115217126*pi,1.5121343228512991*pi) q[13];
U1q(0.212013920634058*pi,0.7385518382677994*pi) q[14];
U1q(0.613712024206395*pi,0.8093805356126005*pi) q[15];
U1q(0.687611907921274*pi,1.125205094808301*pi) q[16];
U1q(0.623045767708201*pi,1.3697012398334003*pi) q[17];
U1q(0.251357104841586*pi,0.9058650636430983*pi) q[18];
U1q(0.392602243460076*pi,1.082625865351801*pi) q[19];
U1q(0.394179608290088*pi,0.5144974756685983*pi) q[20];
U1q(0.873996853681406*pi,0.27414141948429993*pi) q[21];
U1q(0.494556486446918*pi,0.6744612932640983*pi) q[22];
U1q(0.588914473560191*pi,1.4932368119809993*pi) q[23];
U1q(0.380377445621281*pi,1.7594776641497987*pi) q[24];
U1q(0.349892934558743*pi,1.8712281383896805*pi) q[25];
U1q(0.782624028789699*pi,1.4817150488058992*pi) q[26];
U1q(0.528498984471681*pi,1.7533830286942997*pi) q[27];
U1q(0.561426022633272*pi,0.12017117152920065*pi) q[28];
U1q(0.715215199128141*pi,0.2519196563771011*pi) q[29];
U1q(0.422157197952068*pi,0.07544836447160108*pi) q[30];
U1q(0.690160903327047*pi,0.7090401817231999*pi) q[31];
U1q(0.409744028032626*pi,0.6471158224709015*pi) q[32];
U1q(0.229154743419268*pi,1.5022829086635987*pi) q[33];
U1q(0.612982952427101*pi,0.5579215743637995*pi) q[34];
U1q(0.26515327056602*pi,1.2888258767870013*pi) q[35];
U1q(0.364366672332366*pi,1.0520453001436003*pi) q[36];
U1q(0.156685337980782*pi,1.9444405511795004*pi) q[37];
U1q(0.357019770447973*pi,0.34669274674820016*pi) q[38];
U1q(0.341751700930061*pi,0.19570807678720037*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[32];
RZZ(0.5*pi) q[2],q[7];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[11],q[5];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[13],q[34];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[39],q[20];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[35],q[33];
U1q(0.585628024784471*pi,1.0992169762761996*pi) q[0];
U1q(0.421985987037847*pi,0.45049229690770076*pi) q[1];
U1q(0.193191007989599*pi,1.2264541218664995*pi) q[2];
U1q(0.480777979357236*pi,0.01788495093520126*pi) q[3];
U1q(0.0884273482261666*pi,1.1746485282947*pi) q[4];
U1q(0.417917713045674*pi,0.5320204707417986*pi) q[5];
U1q(0.284427348832896*pi,1.0298013409679*pi) q[6];
U1q(0.73450613002682*pi,0.6089125988508002*pi) q[7];
U1q(0.379205945743631*pi,1.6370417442714*pi) q[8];
U1q(0.451905211925164*pi,0.6566441891263004*pi) q[9];
U1q(0.469953068126467*pi,1.981273685427599*pi) q[10];
U1q(0.808648756528338*pi,0.06059036659760153*pi) q[11];
U1q(0.788927675876114*pi,0.22430560982250114*pi) q[12];
U1q(0.127444482233702*pi,1.1926063168167005*pi) q[13];
U1q(0.71786344215037*pi,1.2138649973046007*pi) q[14];
U1q(0.702034729362599*pi,1.2706597013328*pi) q[15];
U1q(0.371805427264201*pi,0.4114706837161002*pi) q[16];
U1q(0.595738980078395*pi,0.6559528511420005*pi) q[17];
U1q(0.674881459727549*pi,1.6981706138702002*pi) q[18];
U1q(0.87399416905331*pi,1.0314598347376993*pi) q[19];
U1q(0.088133641613452*pi,1.6339527225945005*pi) q[20];
U1q(0.71534314939301*pi,0.6971493129279995*pi) q[21];
U1q(0.73786610986267*pi,0.9987910539485014*pi) q[22];
U1q(0.646824907572565*pi,1.4794434581683014*pi) q[23];
U1q(0.278907995114522*pi,1.4480888039983988*pi) q[24];
U1q(0.419398823103713*pi,0.7537499825875997*pi) q[25];
U1q(0.471698167229622*pi,0.3176577353132011*pi) q[26];
U1q(0.0656129878477856*pi,1.0091311438496007*pi) q[27];
U1q(0.242709730120472*pi,0.1137630343531999*pi) q[28];
U1q(0.598457772002637*pi,1.3634700708961986*pi) q[29];
U1q(0.691988692237574*pi,1.763980753545301*pi) q[30];
U1q(0.812233359375525*pi,1.7886067435657012*pi) q[31];
U1q(0.726281247078564*pi,0.463180185663699*pi) q[32];
U1q(0.563949005105637*pi,0.9768802667088003*pi) q[33];
U1q(0.342120015583895*pi,0.1456530535410998*pi) q[34];
U1q(0.806839386433048*pi,1.2786288214349*pi) q[35];
U1q(0.499701879581387*pi,0.7823620643816014*pi) q[36];
U1q(0.678077043341831*pi,1.9129914987397*pi) q[37];
U1q(0.807234208260748*pi,0.6189723827449001*pi) q[38];
U1q(0.797940530096406*pi,1.7173660836342997*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[3],q[34];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[22],q[5];
RZZ(0.5*pi) q[38],q[6];
RZZ(0.5*pi) q[29],q[7];
RZZ(0.5*pi) q[8],q[23];
RZZ(0.5*pi) q[9],q[39];
RZZ(0.5*pi) q[17],q[10];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[14],q[26];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[31],q[24];
RZZ(0.5*pi) q[25],q[28];
U1q(0.12042522092207*pi,0.3944314807772997*pi) q[0];
U1q(0.921280714277592*pi,1.1922409869239985*pi) q[1];
U1q(0.194312683278875*pi,0.37638187614970064*pi) q[2];
U1q(0.591644119017403*pi,1.2585437912993989*pi) q[3];
U1q(0.56204249260533*pi,1.544163760779501*pi) q[4];
U1q(0.090634082640901*pi,1.6587439762021994*pi) q[5];
U1q(0.194970513614994*pi,0.9881432877773015*pi) q[6];
U1q(0.379184683963834*pi,0.1773063176853995*pi) q[7];
U1q(0.558229641538876*pi,0.4053411489181009*pi) q[8];
U1q(0.27496765501748*pi,1.8957179110204017*pi) q[9];
U1q(0.390569087362639*pi,0.6600654654458005*pi) q[10];
U1q(0.633359956324164*pi,0.5043525625643994*pi) q[11];
U1q(0.381621756708839*pi,1.805924788103301*pi) q[12];
U1q(0.657488992279583*pi,0.5741793908734998*pi) q[13];
U1q(0.589013595561126*pi,0.49950647345210086*pi) q[14];
U1q(0.543536138449228*pi,1.4323409036360992*pi) q[15];
U1q(0.0672324538693739*pi,1.9379964383489998*pi) q[16];
U1q(0.197166722307494*pi,0.14567698925739947*pi) q[17];
U1q(0.596568745680894*pi,1.7873409640227997*pi) q[18];
U1q(0.213209625452516*pi,0.0140544384607999*pi) q[19];
U1q(0.841687697287831*pi,0.5234949380798994*pi) q[20];
U1q(0.726735505785536*pi,0.502957015762501*pi) q[21];
U1q(0.39233322078284*pi,1.0235639467513984*pi) q[22];
U1q(0.615729964935873*pi,0.21428027681420048*pi) q[23];
U1q(0.29606518297369*pi,1.3245789816310989*pi) q[24];
U1q(0.501640285588292*pi,1.4808675795698*pi) q[25];
U1q(0.172061686883742*pi,0.5709828381758015*pi) q[26];
U1q(0.571898848013654*pi,0.04757682348610004*pi) q[27];
U1q(0.383201618412476*pi,0.21961168037979917*pi) q[28];
U1q(0.524613170543036*pi,1.6062053413984998*pi) q[29];
U1q(0.571363687858411*pi,0.9775976004230991*pi) q[30];
U1q(0.58855361104213*pi,1.511288075332601*pi) q[31];
U1q(0.424031793353835*pi,1.5730538528424987*pi) q[32];
U1q(0.480066317537412*pi,0.6717911450856988*pi) q[33];
U1q(0.272307520352357*pi,0.08537818437169875*pi) q[34];
U1q(0.359510930863394*pi,0.26575027704990006*pi) q[35];
U1q(0.648711959472972*pi,1.5671036744119995*pi) q[36];
U1q(0.376879612480072*pi,1.9500284745993*pi) q[37];
U1q(0.410963427014938*pi,1.2354428872680998*pi) q[38];
U1q(0.292614542708571*pi,1.553561606812*pi) q[39];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[3],q[20];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[18],q[5];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[25],q[7];
RZZ(0.5*pi) q[8],q[32];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[38],q[22];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[30],q[31];
U1q(0.609374060057939*pi,1.2748730140339006*pi) q[0];
U1q(0.672314040901733*pi,1.0711489988568985*pi) q[1];
U1q(0.242088904402515*pi,0.10396633502569941*pi) q[2];
U1q(0.243624128775688*pi,1.9987542107319989*pi) q[3];
U1q(0.444747198048423*pi,1.5608476732423995*pi) q[4];
U1q(0.958858923902521*pi,0.908582160090301*pi) q[5];
U1q(0.104060403856687*pi,0.24636020814680037*pi) q[6];
U1q(0.502658512000856*pi,0.6397514829758997*pi) q[7];
U1q(0.480889891713888*pi,0.019871416265900166*pi) q[8];
U1q(0.273976080193599*pi,0.2756618497449992*pi) q[9];
U1q(0.566092131660214*pi,0.8906055408981999*pi) q[10];
U1q(0.621026353246009*pi,1.2097235033360008*pi) q[11];
U1q(0.553028712646629*pi,1.1000017489396008*pi) q[12];
U1q(0.149799014993485*pi,1.5059079495149987*pi) q[13];
U1q(0.614405696870924*pi,1.1735995758261986*pi) q[14];
U1q(0.180172778743213*pi,1.7436914844365994*pi) q[15];
U1q(0.356641221413159*pi,1.1024989832576004*pi) q[16];
U1q(0.798433487480632*pi,0.40319762550349836*pi) q[17];
U1q(0.558736858557154*pi,1.6316478636069007*pi) q[18];
U1q(0.776507799378901*pi,1.3499149592479007*pi) q[19];
U1q(0.761468392606712*pi,0.3830534307101985*pi) q[20];
U1q(0.568573357524417*pi,0.23909358836259997*pi) q[21];
U1q(0.860749857226219*pi,1.9976087650486*pi) q[22];
U1q(0.490561710090331*pi,0.08389266673840012*pi) q[23];
U1q(0.843882438349501*pi,1.6490505450416002*pi) q[24];
U1q(0.279698514684913*pi,0.8330868968737004*pi) q[25];
U1q(0.386699392303041*pi,0.6574634276853004*pi) q[26];
U1q(0.226759765957464*pi,0.7186264444136015*pi) q[27];
U1q(0.582866539478549*pi,0.07373458990049997*pi) q[28];
U1q(0.631603922935247*pi,1.9655485496148017*pi) q[29];
U1q(0.270606506839461*pi,1.5382991000326989*pi) q[30];
U1q(0.549740393419099*pi,0.7092260218358*pi) q[31];
U1q(0.794782698644927*pi,0.05755180617889977*pi) q[32];
U1q(0.593216757128199*pi,0.8980756581504004*pi) q[33];
U1q(0.886467021560435*pi,0.169511219660599*pi) q[34];
U1q(0.385398182097498*pi,0.07488203988580011*pi) q[35];
U1q(0.311319854742913*pi,0.8036244125792003*pi) q[36];
U1q(0.670505057956284*pi,1.865059085816899*pi) q[37];
U1q(0.902243139083996*pi,1.5939880661833001*pi) q[38];
U1q(0.205125582777333*pi,1.1180268322884004*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[6],q[4];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[14],q[12];
RZZ(0.5*pi) q[13],q[32];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[27],q[34];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[35],q[36];
U1q(0.570441172682839*pi,0.038960833770200765*pi) q[0];
U1q(0.615080445637039*pi,0.7986408839538015*pi) q[1];
U1q(0.648853215600202*pi,1.6553218770542983*pi) q[2];
U1q(0.447000177875572*pi,0.9456393158689984*pi) q[3];
U1q(0.346998452718266*pi,0.8513712417050989*pi) q[4];
U1q(0.573256848441064*pi,0.8254988666558987*pi) q[5];
U1q(0.660901793711753*pi,1.9065460837595012*pi) q[6];
U1q(0.190417522683902*pi,1.4395935109529994*pi) q[7];
U1q(0.451041172148839*pi,0.09580743022520011*pi) q[8];
U1q(0.480999791644408*pi,0.9650378369035018*pi) q[9];
U1q(0.467863344392102*pi,1.3198642828999994*pi) q[10];
U1q(0.671411974099931*pi,0.07371166706710142*pi) q[11];
U1q(0.304511689003419*pi,0.6296930115383006*pi) q[12];
U1q(0.789563432091486*pi,1.3265888881566*pi) q[13];
U1q(0.502839251494592*pi,1.8639072935539005*pi) q[14];
U1q(0.701847390783275*pi,0.29527991229450024*pi) q[15];
U1q(0.585139649618024*pi,1.2632440791854016*pi) q[16];
U1q(0.781615321247985*pi,0.5729697820228985*pi) q[17];
U1q(0.568204800069876*pi,0.42652900265759897*pi) q[18];
U1q(0.413533333602122*pi,0.5270011447790992*pi) q[19];
U1q(0.794152637994813*pi,0.6129129412681991*pi) q[20];
U1q(0.623256781193215*pi,0.3070366900178989*pi) q[21];
U1q(0.509748141463996*pi,0.6958287680104007*pi) q[22];
U1q(0.255673479873703*pi,1.0640429373836007*pi) q[23];
U1q(0.786531389871969*pi,1.338770580450099*pi) q[24];
U1q(0.717893519162814*pi,0.8200440197511991*pi) q[25];
U1q(0.365105603277185*pi,1.0435725422403017*pi) q[26];
U1q(0.185980172197171*pi,1.7240579535802993*pi) q[27];
U1q(0.832410237752979*pi,1.0297162483762001*pi) q[28];
U1q(0.663569193190351*pi,1.5074461651752031*pi) q[29];
U1q(0.637558044692205*pi,1.8052347280818992*pi) q[30];
U1q(0.504148014708418*pi,0.46874299554960075*pi) q[31];
U1q(0.592321555183961*pi,0.12737635570659833*pi) q[32];
U1q(0.772307130419494*pi,1.4315544045278017*pi) q[33];
U1q(0.469527391560793*pi,0.42738587808510076*pi) q[34];
U1q(0.618242119070079*pi,1.5885075979631011*pi) q[35];
U1q(0.408682911489848*pi,1.8603653122947001*pi) q[36];
U1q(0.695568026921525*pi,0.9490962535144014*pi) q[37];
U1q(0.545126033960832*pi,0.8108770425488991*pi) q[38];
U1q(0.671499187211662*pi,1.5763610148175005*pi) q[39];
rz(3.8737715981575*pi) q[0];
rz(1.0677224040540985*pi) q[1];
rz(0.32959976699810056*pi) q[2];
rz(3.535280616749599*pi) q[3];
rz(2.2080270903745003*pi) q[4];
rz(2.3664042434116013*pi) q[5];
rz(2.249284328292301*pi) q[6];
rz(2.5665854498263982*pi) q[7];
rz(3.177660212682401*pi) q[8];
rz(2.1272746713255017*pi) q[9];
rz(2.552379050553899*pi) q[10];
rz(2.889292790536601*pi) q[11];
rz(3.3989443313266996*pi) q[12];
rz(3.7300760332771006*pi) q[13];
rz(1.942700062801201*pi) q[14];
rz(2.7380964042732003*pi) q[15];
rz(0.6173139071396001*pi) q[16];
rz(3.0012393372736987*pi) q[17];
rz(2.4690991184285984*pi) q[18];
rz(3.5140612583275015*pi) q[19];
rz(3.9275890435567007*pi) q[20];
rz(3.130455398462299*pi) q[21];
rz(3.923246027573999*pi) q[22];
rz(2.5509576950693997*pi) q[23];
rz(1.8604850471297993*pi) q[24];
rz(3.3903866288189004*pi) q[25];
rz(3.201720665709999*pi) q[26];
rz(0.6613367041530012*pi) q[27];
rz(0.6629995999587983*pi) q[28];
rz(0.22642435422360307*pi) q[29];
rz(1.6652383695196988*pi) q[30];
rz(2.7147202137856006*pi) q[31];
rz(2.739913272421301*pi) q[32];
rz(1.7809733437000013*pi) q[33];
rz(1.7669172706914011*pi) q[34];
rz(0.5080971797034977*pi) q[35];
rz(3.8456975340382016*pi) q[36];
rz(1.8050622619507983*pi) q[37];
rz(1.0772854010687993*pi) q[38];
rz(3.7851406594591985*pi) q[39];
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