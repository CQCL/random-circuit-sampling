OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.528897277905798*pi,1.560601675724115*pi) q[0];
U1q(0.272738064937346*pi,1.3148651453523739*pi) q[1];
U1q(0.516857234774036*pi,0.00362634074738011*pi) q[2];
U1q(0.506469968735047*pi,1.22866656982846*pi) q[3];
U1q(0.457877796076396*pi,1.450157168838865*pi) q[4];
U1q(0.542919796227984*pi,0.870387666317376*pi) q[5];
U1q(0.319289886703824*pi,0.201875941461625*pi) q[6];
U1q(0.871912223948323*pi,0.304169096197867*pi) q[7];
U1q(0.335254392507963*pi,0.392619543031974*pi) q[8];
U1q(0.349408292446811*pi,0.8857043860593501*pi) q[9];
U1q(0.516365020322655*pi,1.19099568500977*pi) q[10];
U1q(0.181076679806305*pi,1.0202670195622239*pi) q[11];
U1q(0.474118804841928*pi,0.774248458493358*pi) q[12];
U1q(0.131773382164008*pi,1.707658715100193*pi) q[13];
U1q(0.536307653815307*pi,0.450453486698977*pi) q[14];
U1q(0.404491845697071*pi,1.405982360123713*pi) q[15];
U1q(0.233707427442121*pi,0.313012799960923*pi) q[16];
U1q(0.562031869455504*pi,0.783262593625929*pi) q[17];
U1q(0.309819878707423*pi,1.615199721665306*pi) q[18];
U1q(0.382338728000853*pi,1.396965304184247*pi) q[19];
U1q(0.507335498613288*pi,1.314516510489978*pi) q[20];
U1q(0.773899090553381*pi,1.9498730547549599*pi) q[21];
U1q(0.206349176135669*pi,1.5067823840270749*pi) q[22];
U1q(0.647817450808408*pi,0.607290486687671*pi) q[23];
U1q(0.148066497628102*pi,1.366722898008925*pi) q[24];
U1q(0.18390541926435*pi,0.438716870227973*pi) q[25];
U1q(0.730453574976295*pi,1.66288236279787*pi) q[26];
U1q(0.469633620274762*pi,0.500581189533814*pi) q[27];
U1q(0.291737918317205*pi,1.9925899422952797*pi) q[28];
U1q(0.441530633583296*pi,0.696401712515862*pi) q[29];
U1q(0.425380471178156*pi,0.9285728114205101*pi) q[30];
U1q(0.150314276954078*pi,1.9167129006441446*pi) q[31];
U1q(0.59564251987667*pi,1.461771834269114*pi) q[32];
U1q(0.517065355230455*pi,0.361744201909362*pi) q[33];
U1q(0.532867147704558*pi,1.816103855932949*pi) q[34];
U1q(0.24676513406398*pi,0.392789530975061*pi) q[35];
U1q(0.424885829322373*pi,1.834716347172443*pi) q[36];
U1q(0.884976126110315*pi,1.882456519395392*pi) q[37];
U1q(0.656247661796446*pi,0.98287796649025*pi) q[38];
U1q(0.220389947524831*pi,1.632604512992068*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[28],q[8];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[31],q[10];
RZZ(0.5*pi) q[29],q[11];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[19],q[14];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[39],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[33],q[38];
U1q(0.42331848224168*pi,1.4150329180331198*pi) q[0];
U1q(0.176248231532911*pi,1.1340326985506*pi) q[1];
U1q(0.590810353993047*pi,1.7645835545687798*pi) q[2];
U1q(0.288936952146739*pi,1.9288192828112*pi) q[3];
U1q(0.721670096995338*pi,1.27778587433124*pi) q[4];
U1q(0.49390188908983*pi,1.453568327398781*pi) q[5];
U1q(0.693143141786215*pi,1.97283947988713*pi) q[6];
U1q(0.383317524933459*pi,0.1568709496760301*pi) q[7];
U1q(0.311473905893649*pi,0.14072092022616012*pi) q[8];
U1q(0.602873022472491*pi,0.37305062618018003*pi) q[9];
U1q(0.177349929008981*pi,0.9269635591946399*pi) q[10];
U1q(0.22566641590728*pi,1.2219992966716902*pi) q[11];
U1q(0.806173674119296*pi,1.47547741017969*pi) q[12];
U1q(0.342491700646551*pi,1.41193346686086*pi) q[13];
U1q(0.085361342609796*pi,0.7237269889152498*pi) q[14];
U1q(0.24565882227397*pi,1.54271653260888*pi) q[15];
U1q(0.420550070544912*pi,0.9855262841100498*pi) q[16];
U1q(0.546480077679675*pi,1.5456093484538869*pi) q[17];
U1q(0.697703405334624*pi,1.37713931719728*pi) q[18];
U1q(0.924910325508596*pi,0.6594361615641802*pi) q[19];
U1q(0.62384351919223*pi,1.8440968635577697*pi) q[20];
U1q(0.559777350454062*pi,0.037452166021029853*pi) q[21];
U1q(0.388093274325695*pi,1.0963108450889498*pi) q[22];
U1q(0.237953141476148*pi,1.350923355594528*pi) q[23];
U1q(0.509246298701233*pi,0.8368750186946898*pi) q[24];
U1q(0.54376789078222*pi,1.251851860555192*pi) q[25];
U1q(0.430136582893849*pi,1.7884875878139503*pi) q[26];
U1q(0.588660548284741*pi,1.7904463930417198*pi) q[27];
U1q(0.47291698472879*pi,0.10018885422525003*pi) q[28];
U1q(0.390151440386431*pi,0.9222977343634802*pi) q[29];
U1q(0.469523408313101*pi,1.7429946392569997*pi) q[30];
U1q(0.182251512574832*pi,1.8279358693265602*pi) q[31];
U1q(0.699372864352023*pi,0.42228229578706*pi) q[32];
U1q(0.77911397039624*pi,0.1310294488580801*pi) q[33];
U1q(0.274734866000792*pi,0.5059169686116198*pi) q[34];
U1q(0.904905087417973*pi,1.6275517365383698*pi) q[35];
U1q(0.595392000999746*pi,1.8758821976087496*pi) q[36];
U1q(0.894632706080163*pi,1.93300227927632*pi) q[37];
U1q(0.644133838004343*pi,1.7750335743387802*pi) q[38];
U1q(0.131711282922056*pi,0.10525036184861003*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[38],q[12];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[33],q[29];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[34],q[36];
U1q(0.685968725576045*pi,0.46270354472125996*pi) q[0];
U1q(0.39126573674028*pi,0.3284266969054297*pi) q[1];
U1q(0.478138288444612*pi,1.20619167743423*pi) q[2];
U1q(0.24006513674247*pi,1.9895001124486003*pi) q[3];
U1q(0.762649235737539*pi,1.70053597838442*pi) q[4];
U1q(0.339063481068431*pi,0.9819467887535098*pi) q[5];
U1q(0.656456874558656*pi,0.45532520213340977*pi) q[6];
U1q(0.587192880104281*pi,1.41443324476948*pi) q[7];
U1q(0.686834363307752*pi,0.24217066579474977*pi) q[8];
U1q(0.446273003293469*pi,1.8384565366538599*pi) q[9];
U1q(0.312280945703995*pi,1.9156292801825696*pi) q[10];
U1q(0.684848540046326*pi,1.1861335354362303*pi) q[11];
U1q(0.511431064322761*pi,0.4737674164482*pi) q[12];
U1q(0.789979239578749*pi,1.0916181526845996*pi) q[13];
U1q(0.29442451261557*pi,0.5023641892115602*pi) q[14];
U1q(0.436626345558141*pi,1.9197931380994104*pi) q[15];
U1q(0.614895436680787*pi,1.0202746997902903*pi) q[16];
U1q(0.0265758865273605*pi,1.38551856965735*pi) q[17];
U1q(0.808847337737915*pi,1.7737549717656398*pi) q[18];
U1q(0.111692483792162*pi,1.2221481434319896*pi) q[19];
U1q(0.585029711789491*pi,0.08319269263126028*pi) q[20];
U1q(0.730788581707565*pi,1.3195255580958198*pi) q[21];
U1q(0.371726951733562*pi,1.7108528066895499*pi) q[22];
U1q(0.107505866456615*pi,0.6439841602049099*pi) q[23];
U1q(0.106255182586141*pi,1.2046824936310898*pi) q[24];
U1q(0.168657655256709*pi,0.9519329227609701*pi) q[25];
U1q(0.517677928227657*pi,0.52631242987876*pi) q[26];
U1q(0.749721677200659*pi,0.3049346798543997*pi) q[27];
U1q(0.97502639308806*pi,0.5813411929758203*pi) q[28];
U1q(0.86972514850959*pi,1.92495912614475*pi) q[29];
U1q(0.186384623004684*pi,0.8496242873798501*pi) q[30];
U1q(0.811312035123978*pi,1.77698968283726*pi) q[31];
U1q(0.628651452584094*pi,1.14519193467775*pi) q[32];
U1q(0.19780522994061*pi,0.8739349469354099*pi) q[33];
U1q(0.264621655927516*pi,1.0028692571047904*pi) q[34];
U1q(0.136946373491*pi,1.0134726877944402*pi) q[35];
U1q(0.124155143009506*pi,0.3913721223630997*pi) q[36];
U1q(0.49472362420044*pi,1.9254393073632903*pi) q[37];
U1q(0.250271805452347*pi,1.3016616846415197*pi) q[38];
U1q(0.847227915232238*pi,1.3709077729393897*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[29],q[10];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[31],q[24];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[32],q[35];
RZZ(0.5*pi) q[33],q[34];
RZZ(0.5*pi) q[37],q[39];
U1q(0.804355958859261*pi,1.0060379926431704*pi) q[0];
U1q(0.402515980681376*pi,0.14653012290009038*pi) q[1];
U1q(0.700956697608139*pi,0.06254232808070981*pi) q[2];
U1q(0.918897197698472*pi,0.29959580517930995*pi) q[3];
U1q(0.570248483389246*pi,0.9797783335222299*pi) q[4];
U1q(0.334514357215315*pi,0.6050973998024798*pi) q[5];
U1q(0.0354953255217616*pi,0.68100337212773*pi) q[6];
U1q(0.425101729824728*pi,0.7527091248466498*pi) q[7];
U1q(0.58658403452955*pi,1.45222752384311*pi) q[8];
U1q(0.323872718400614*pi,1.9394781031787698*pi) q[9];
U1q(0.23063577582938*pi,1.3299821349663796*pi) q[10];
U1q(0.422218451398856*pi,0.38105174344091*pi) q[11];
U1q(0.771268472298131*pi,0.8070118510168696*pi) q[12];
U1q(0.305127787682059*pi,0.6299223165990897*pi) q[13];
U1q(0.159025035560161*pi,0.04654360235919963*pi) q[14];
U1q(0.127032311559068*pi,1.78776469115512*pi) q[15];
U1q(0.508709744263266*pi,0.9707716528853396*pi) q[16];
U1q(0.493152565644012*pi,0.2907630806364603*pi) q[17];
U1q(0.691266379443487*pi,1.3271563508113102*pi) q[18];
U1q(0.43134899679382*pi,1.1788211797188808*pi) q[19];
U1q(0.527233585231169*pi,0.5512336412331598*pi) q[20];
U1q(0.532311825136937*pi,0.8708172724775594*pi) q[21];
U1q(0.664353003369633*pi,0.5688151282023002*pi) q[22];
U1q(0.342554977805118*pi,0.8503076830988601*pi) q[23];
U1q(0.910509131979087*pi,1.0663117831874107*pi) q[24];
U1q(0.389801264838848*pi,0.8523519632475098*pi) q[25];
U1q(0.0664417767039785*pi,0.5544480455429497*pi) q[26];
U1q(0.737120701183654*pi,0.67831921032277*pi) q[27];
U1q(0.533062527995358*pi,0.27197783052633007*pi) q[28];
U1q(0.196509014291537*pi,1.7397054655821398*pi) q[29];
U1q(0.460226370909558*pi,0.9765543967693002*pi) q[30];
U1q(0.141574890797827*pi,1.9998352614080002*pi) q[31];
U1q(0.593234079699527*pi,0.9384844768906002*pi) q[32];
U1q(0.371556873213729*pi,1.1024297266409802*pi) q[33];
U1q(0.312917545453036*pi,0.7006364413965702*pi) q[34];
U1q(0.852131348429747*pi,0.004036139709129749*pi) q[35];
U1q(0.228904701533704*pi,0.6302233279921996*pi) q[36];
U1q(0.527210305509938*pi,0.7747853332535302*pi) q[37];
U1q(0.231099642950451*pi,0.6454811552031501*pi) q[38];
U1q(0.680362821029616*pi,1.9090545299748998*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[27],q[2];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[36];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[28],q[29];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[32],q[37];
U1q(0.529420353549916*pi,1.7010023932431704*pi) q[0];
U1q(0.542397254661018*pi,1.5594289363139993*pi) q[1];
U1q(0.661284276444843*pi,0.20891010835632073*pi) q[2];
U1q(0.133118738605364*pi,1.17855193352867*pi) q[3];
U1q(0.163193972063904*pi,1.1394020578653006*pi) q[4];
U1q(0.312012192059293*pi,1.7137237331668604*pi) q[5];
U1q(0.77878118578526*pi,1.4511070495652003*pi) q[6];
U1q(0.37253283767209*pi,0.9839065471085302*pi) q[7];
U1q(0.688615737816807*pi,1.7183127859858995*pi) q[8];
U1q(0.463587828052505*pi,1.7574975928247891*pi) q[9];
U1q(0.376983192307621*pi,0.13750571111955967*pi) q[10];
U1q(0.644595752634039*pi,1.1564977858063994*pi) q[11];
U1q(0.100676482357583*pi,1.4018898299528004*pi) q[12];
U1q(0.525938040270042*pi,0.34744677479211994*pi) q[13];
U1q(0.483534832454155*pi,0.22325939126564975*pi) q[14];
U1q(0.368724972354497*pi,1.9981312142445997*pi) q[15];
U1q(0.438390921178257*pi,1.0530195941016505*pi) q[16];
U1q(0.0871844907921216*pi,1.8634592757021995*pi) q[17];
U1q(0.575470528799387*pi,1.4075194861830909*pi) q[18];
U1q(0.226076696865666*pi,0.39945577562249923*pi) q[19];
U1q(0.389403634231226*pi,1.9196273317622996*pi) q[20];
U1q(0.980325841546714*pi,0.9114757802482991*pi) q[21];
U1q(0.583298095362728*pi,0.5144647718565896*pi) q[22];
U1q(0.168550668742704*pi,0.9686310128854796*pi) q[23];
U1q(0.486077656247999*pi,0.22776476228940012*pi) q[24];
U1q(0.654963408518399*pi,0.3317831740713997*pi) q[25];
U1q(0.761301418856113*pi,1.8107975225688993*pi) q[26];
U1q(0.121242818616093*pi,0.03732331449995918*pi) q[27];
U1q(0.302057916000011*pi,1.5032823526466998*pi) q[28];
U1q(0.784089591435564*pi,0.4237130903142905*pi) q[29];
U1q(0.318174050990323*pi,1.5388772198616998*pi) q[30];
U1q(0.284921039094207*pi,1.2698480117649993*pi) q[31];
U1q(0.583198203448922*pi,1.2002993494227*pi) q[32];
U1q(0.825496155705783*pi,1.8812266856627993*pi) q[33];
U1q(0.774829277697763*pi,1.0425349682104006*pi) q[34];
U1q(0.824347942018376*pi,0.23323814610713*pi) q[35];
U1q(0.552956234352063*pi,0.7303424288685996*pi) q[36];
U1q(0.732350389249172*pi,1.2685039659838893*pi) q[37];
U1q(0.491873957324472*pi,1.0230230291918705*pi) q[38];
U1q(0.603259156755172*pi,1.5329801246931396*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[5],q[36];
RZZ(0.5*pi) q[16],q[6];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[23],q[8];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[30],q[12];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[26],q[17];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[34],q[29];
RZZ(0.5*pi) q[32],q[38];
U1q(0.697966059201654*pi,0.4562679139881105*pi) q[0];
U1q(0.551335805626356*pi,0.30408770556459963*pi) q[1];
U1q(0.118105644909892*pi,1.4906432615618996*pi) q[2];
U1q(0.849902033352022*pi,0.5630241489174992*pi) q[3];
U1q(0.348092458097171*pi,0.9507230282654007*pi) q[4];
U1q(0.445329187057889*pi,1.4146818265608*pi) q[5];
U1q(0.308870473531698*pi,1.9787000326266995*pi) q[6];
U1q(0.294234393654077*pi,0.5270977127098*pi) q[7];
U1q(0.404497175847764*pi,1.3511081845097*pi) q[8];
U1q(0.620757770449544*pi,0.7091193160410008*pi) q[9];
U1q(0.406865813277185*pi,1.3750357984150003*pi) q[10];
U1q(0.7565895802425*pi,1.6918669676368996*pi) q[11];
U1q(0.797061001545261*pi,0.07889969765722071*pi) q[12];
U1q(0.134229927912015*pi,1.0609252277509995*pi) q[13];
U1q(0.642982983977421*pi,0.45944687866069955*pi) q[14];
U1q(0.475824979523557*pi,1.3505890290146993*pi) q[15];
U1q(0.13484663840866*pi,0.34459876180910065*pi) q[16];
U1q(0.295459134614747*pi,0.5779222164597009*pi) q[17];
U1q(0.556985465531599*pi,0.3956308707252294*pi) q[18];
U1q(0.660609206233293*pi,0.8964497611430993*pi) q[19];
U1q(0.324501547916075*pi,1.8762741864508996*pi) q[20];
U1q(0.599596439482467*pi,1.6930216863016003*pi) q[21];
U1q(0.618376445426884*pi,0.14640157496170048*pi) q[22];
U1q(0.499815709681819*pi,1.0884911143307896*pi) q[23];
U1q(0.618188498487916*pi,1.8889384265327003*pi) q[24];
U1q(0.595348502582014*pi,1.1139055453730604*pi) q[25];
U1q(0.618782052012638*pi,1.7443584230603992*pi) q[26];
U1q(0.484691265796004*pi,0.7983850733668003*pi) q[27];
U1q(0.799042781363671*pi,0.22833649940839962*pi) q[28];
U1q(0.250990531345835*pi,0.6326044355212996*pi) q[29];
U1q(0.381776123742049*pi,1.7883885207738004*pi) q[30];
U1q(0.665250785648834*pi,1.4894504442015002*pi) q[31];
U1q(0.319656775600784*pi,0.19947612640879964*pi) q[32];
U1q(0.602992058853024*pi,0.8397154319772007*pi) q[33];
U1q(0.376606829739315*pi,1.7129744770361004*pi) q[34];
U1q(0.0446650730324477*pi,1.3720148903361*pi) q[35];
U1q(0.396547289307573*pi,0.9662353230249998*pi) q[36];
U1q(0.611497526332597*pi,0.41066361931459916*pi) q[37];
U1q(0.414733033207744*pi,1.8608999723383999*pi) q[38];
U1q(0.588774166648927*pi,1.4801123704621801*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[31];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[16],q[14];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[19],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[34],q[20];
RZZ(0.5*pi) q[21],q[36];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[26],q[28];
U1q(0.245949951449187*pi,1.6895616513820002*pi) q[0];
U1q(0.425016249003899*pi,1.5103830056339014*pi) q[1];
U1q(0.518704486259505*pi,0.21437612431500064*pi) q[2];
U1q(0.684200826444178*pi,0.061067765043999245*pi) q[3];
U1q(0.557632722797274*pi,0.2935961415294006*pi) q[4];
U1q(0.241651436131711*pi,1.8014626595566003*pi) q[5];
U1q(0.858779958483495*pi,1.6952627658519006*pi) q[6];
U1q(0.762475893815248*pi,1.3134755643812994*pi) q[7];
U1q(0.793676262242971*pi,1.9520108400394989*pi) q[8];
U1q(0.406391316283707*pi,1.0487961022905008*pi) q[9];
U1q(0.710640365691053*pi,0.1828851960289004*pi) q[10];
U1q(0.565806431885714*pi,1.1225619523742*pi) q[11];
U1q(0.760684053846144*pi,0.37880347750230037*pi) q[12];
U1q(0.68039215987359*pi,0.4045841147763003*pi) q[13];
U1q(0.582810992714733*pi,1.5371678353505*pi) q[14];
U1q(0.828965548520145*pi,1.3319041098683009*pi) q[15];
U1q(0.843224942217952*pi,0.7768911446610005*pi) q[16];
U1q(0.33490658546919*pi,0.9307749654172994*pi) q[17];
U1q(0.215062659204652*pi,0.5529781589288003*pi) q[18];
U1q(0.0843397908264772*pi,1.2096668366923993*pi) q[19];
U1q(0.0493625576734576*pi,1.4657420351812007*pi) q[20];
U1q(0.736874263384154*pi,1.8687978725921006*pi) q[21];
U1q(0.0306070772199901*pi,1.2006579966858002*pi) q[22];
U1q(0.580467022114565*pi,1.3409203185037004*pi) q[23];
U1q(0.894655857246236*pi,0.9348755775961006*pi) q[24];
U1q(0.0545202579169734*pi,0.8316647257006107*pi) q[25];
U1q(0.319575872738574*pi,0.45766095937580076*pi) q[26];
U1q(0.483556993144443*pi,1.0101248207083984*pi) q[27];
U1q(0.316435393319812*pi,0.35005864206880055*pi) q[28];
U1q(0.662625211140647*pi,1.6115855598122018*pi) q[29];
U1q(0.838816798713771*pi,1.1461244364541017*pi) q[30];
U1q(0.195939202484904*pi,0.7254882082308995*pi) q[31];
U1q(0.600058747131389*pi,0.8635848821672596*pi) q[32];
U1q(0.530828643586612*pi,0.46228945383050046*pi) q[33];
U1q(0.781034499880253*pi,0.5790432383564017*pi) q[34];
U1q(0.648063285317815*pi,0.2669885059648003*pi) q[35];
U1q(0.628620587523362*pi,0.37835765287019996*pi) q[36];
U1q(0.344394411051547*pi,0.7046547517002999*pi) q[37];
U1q(0.555299145109042*pi,1.4342814083857007*pi) q[38];
U1q(0.190558683616988*pi,1.2552005282901*pi) q[39];
RZZ(0.5*pi) q[0],q[22];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[8],q[24];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[19],q[16];
RZZ(0.5*pi) q[37],q[17];
RZZ(0.5*pi) q[30],q[20];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[23],q[29];
RZZ(0.5*pi) q[38],q[27];
RZZ(0.5*pi) q[32],q[34];
U1q(0.2553212431557*pi,0.28999850368690083*pi) q[0];
U1q(0.830370578475618*pi,1.1914503198242983*pi) q[1];
U1q(0.383988380643742*pi,0.23366410495880174*pi) q[2];
U1q(0.186200941076236*pi,1.2890279612123017*pi) q[3];
U1q(0.682652023383384*pi,0.06384109632049828*pi) q[4];
U1q(0.439841709450786*pi,0.8987510400102998*pi) q[5];
U1q(0.454048665567925*pi,1.8305267223287984*pi) q[6];
U1q(0.249404821477521*pi,0.41953457317049825*pi) q[7];
U1q(0.883282530289959*pi,0.0029170723272997634*pi) q[8];
U1q(0.478002433172006*pi,1.7484358566877987*pi) q[9];
U1q(0.341010780838444*pi,1.9133834746512015*pi) q[10];
U1q(0.543459171212643*pi,1.723421345519899*pi) q[11];
U1q(0.257291560211843*pi,0.13980179950880078*pi) q[12];
U1q(0.37721991927125*pi,1.825002751314699*pi) q[13];
U1q(0.319183075998631*pi,0.48589227115980016*pi) q[14];
U1q(0.335743570472777*pi,0.9225623016293003*pi) q[15];
U1q(0.573149225135202*pi,1.6088044454485004*pi) q[16];
U1q(0.0896583550876081*pi,1.8408881703341002*pi) q[17];
U1q(0.53342206668058*pi,1.4091665204062007*pi) q[18];
U1q(0.683821959133145*pi,0.7844850438732998*pi) q[19];
U1q(0.637903002997066*pi,0.39006554912739944*pi) q[20];
U1q(0.527681476471213*pi,0.5782671758082998*pi) q[21];
U1q(0.420803360924623*pi,1.6079457899249014*pi) q[22];
U1q(0.62636226130918*pi,0.8135878099406*pi) q[23];
U1q(0.160004730102077*pi,0.7186161720578994*pi) q[24];
U1q(0.536152654120798*pi,0.011720536897399825*pi) q[25];
U1q(0.606052141902263*pi,1.897515167835099*pi) q[26];
U1q(0.642006950969596*pi,0.6926988223275998*pi) q[27];
U1q(0.282624272310566*pi,4.188742170008197e-05*pi) q[28];
U1q(0.92707160871683*pi,0.04439686446569979*pi) q[29];
U1q(0.337414788909253*pi,0.15155943543300054*pi) q[30];
U1q(0.113589951699573*pi,0.2981977679121002*pi) q[31];
U1q(0.909891936512522*pi,1.7455806514426992*pi) q[32];
U1q(0.643915957788421*pi,1.1546690167554985*pi) q[33];
U1q(0.111676721602317*pi,1.5099890471913007*pi) q[34];
U1q(0.426745564056405*pi,0.27628170901000004*pi) q[35];
U1q(0.848885311638949*pi,1.5972742150463013*pi) q[36];
U1q(0.712964171742608*pi,0.9203331053229995*pi) q[37];
U1q(0.89666829082815*pi,1.0975666089312988*pi) q[38];
U1q(0.894876109321744*pi,1.3011954362957*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[35],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[3],q[39];
RZZ(0.5*pi) q[31],q[4];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[8],q[14];
RZZ(0.5*pi) q[9],q[10];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[28],q[22];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[38],q[25];
RZZ(0.5*pi) q[30],q[27];
U1q(0.345563523678881*pi,1.4484735319386992*pi) q[0];
U1q(0.407405650447558*pi,0.8391622183857983*pi) q[1];
U1q(0.803563399615125*pi,0.4015970941791984*pi) q[2];
U1q(0.186525545134014*pi,0.8495458481415987*pi) q[3];
U1q(0.379052853313092*pi,1.6987644601320007*pi) q[4];
U1q(0.697511239423096*pi,1.9339171585607993*pi) q[5];
U1q(0.76066247724282*pi,1.4134013709672004*pi) q[6];
U1q(0.80960876860633*pi,0.5821056585534983*pi) q[7];
U1q(0.209464534892802*pi,1.2023865390297992*pi) q[8];
U1q(0.367010171466826*pi,1.1786156720715013*pi) q[9];
U1q(0.316487867278959*pi,0.8751715449038997*pi) q[10];
U1q(0.685131592312576*pi,0.4821035520202983*pi) q[11];
U1q(0.64550587582729*pi,1.8624193800930016*pi) q[12];
U1q(0.680421703480268*pi,0.8807878822880006*pi) q[13];
U1q(0.656367883504013*pi,1.3392896004845003*pi) q[14];
U1q(0.742870681120499*pi,1.5847056269942001*pi) q[15];
U1q(0.690039082613279*pi,1.1554618895710007*pi) q[16];
U1q(0.485104740071852*pi,1.7511277262624994*pi) q[17];
U1q(0.387020484236829*pi,1.1425784916634*pi) q[18];
U1q(0.408759352583445*pi,1.5893936784455995*pi) q[19];
U1q(0.713279088911104*pi,1.5899341482177007*pi) q[20];
U1q(0.394759138730549*pi,1.5886370371473006*pi) q[21];
U1q(0.224989498669*pi,0.823288217940199*pi) q[22];
U1q(0.568924401445519*pi,0.4649372239807992*pi) q[23];
U1q(0.0887441316883192*pi,1.9398333300960005*pi) q[24];
U1q(0.463304282450149*pi,0.2644997170771006*pi) q[25];
U1q(0.761588558416155*pi,1.7175594788760016*pi) q[26];
U1q(0.902210745078007*pi,0.21651882707750048*pi) q[27];
U1q(0.336982362730422*pi,0.4173383202465004*pi) q[28];
U1q(0.529948883528714*pi,1.7752075070727003*pi) q[29];
U1q(0.85793527068623*pi,0.04444774205159874*pi) q[30];
U1q(0.063730311323899*pi,0.9868908584208995*pi) q[31];
U1q(0.219346692430369*pi,1.3644757697447005*pi) q[32];
U1q(0.378147923748683*pi,1.1365499781715016*pi) q[33];
U1q(0.593466428976223*pi,1.2156717997259001*pi) q[34];
U1q(0.381763068761056*pi,1.326155541295801*pi) q[35];
U1q(0.471809198008906*pi,1.3119682829264008*pi) q[36];
U1q(0.13333801549927*pi,1.0534766793753008*pi) q[37];
U1q(0.639375007245998*pi,0.9333265536279995*pi) q[38];
U1q(0.678027386236855*pi,1.7298560875433004*pi) q[39];
rz(2.395199677878299*pi) q[0];
rz(3.041107903817501*pi) q[1];
rz(2.9754811671739*pi) q[2];
rz(3.4451786992684994*pi) q[3];
rz(3.1884927834951*pi) q[4];
rz(1.2149107115291997*pi) q[5];
rz(3.2249385759930007*pi) q[6];
rz(3.258460400366701*pi) q[7];
rz(3.8583598191648*pi) q[8];
rz(3.1437553137416003*pi) q[9];
rz(2.7054808151598984*pi) q[10];
rz(2.2678194953762016*pi) q[11];
rz(1.418514954055599*pi) q[12];
rz(3.1723491052356003*pi) q[13];
rz(2.9131234287034005*pi) q[14];
rz(3.525090208455101*pi) q[15];
rz(1.578626324112001*pi) q[16];
rz(1.0337706839202987*pi) q[17];
rz(2.0783789882259*pi) q[18];
rz(1.502145026236601*pi) q[19];
rz(0.9466925737035012*pi) q[20];
rz(3.7703329148463*pi) q[21];
rz(1.7590721117983996*pi) q[22];
rz(2.1845937922663*pi) q[23];
rz(3.1061111044422987*pi) q[24];
rz(3.8169680059006*pi) q[25];
rz(0.8890729829697008*pi) q[26];
rz(2.5733681814563987*pi) q[27];
rz(0.40882107209040086*pi) q[28];
rz(1.4271101453238018*pi) q[29];
rz(3.8790826249462995*pi) q[30];
rz(0.2912985062345008*pi) q[31];
rz(3.3081843935341*pi) q[32];
rz(2.5876572559836006*pi) q[33];
rz(0.12318866430729969*pi) q[34];
rz(2.4733775652020995*pi) q[35];
rz(2.179069175539201*pi) q[36];
rz(3.5971959761291004*pi) q[37];
rz(0.5045015890467006*pi) q[38];
rz(2.1843640206939003*pi) q[39];
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