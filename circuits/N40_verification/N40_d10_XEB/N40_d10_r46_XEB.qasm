OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.412896144377792*pi,1.824723485842635*pi) q[0];
U1q(0.235725628408581*pi,1.917578754581389*pi) q[1];
U1q(0.498235802580627*pi,0.709543830004926*pi) q[2];
U1q(0.36069183611937*pi,0.0786608296010965*pi) q[3];
U1q(0.477899532329429*pi,0.485683114863603*pi) q[4];
U1q(0.672489857250608*pi,0.529755370312604*pi) q[5];
U1q(0.782132048248549*pi,1.504669249177128*pi) q[6];
U1q(0.858861801708912*pi,1.01771266973106*pi) q[7];
U1q(0.304597644719904*pi,0.921980116266969*pi) q[8];
U1q(0.151633448742027*pi,0.84463494210762*pi) q[9];
U1q(0.361765173566006*pi,1.3133677243025978*pi) q[10];
U1q(0.499501250986562*pi,0.782996651626168*pi) q[11];
U1q(0.127850677522416*pi,1.9854779327518242*pi) q[12];
U1q(0.444033133855994*pi,1.21222085894676*pi) q[13];
U1q(0.507163532284631*pi,0.339131101671335*pi) q[14];
U1q(0.341723607669662*pi,0.922128116285389*pi) q[15];
U1q(0.805997767029118*pi,0.495382163774171*pi) q[16];
U1q(0.206380630512594*pi,0.846643048541753*pi) q[17];
U1q(0.441942220699978*pi,0.05684658575367*pi) q[18];
U1q(0.552985961251808*pi,0.617194865324114*pi) q[19];
U1q(0.204581241011601*pi,0.605430328163789*pi) q[20];
U1q(0.24867874534909*pi,1.699056131209521*pi) q[21];
U1q(0.67702830978437*pi,1.26845856156906*pi) q[22];
U1q(0.537609905112858*pi,0.646043995416808*pi) q[23];
U1q(0.340748126762616*pi,0.555212473248716*pi) q[24];
U1q(0.5431163621506*pi,1.759702428671851*pi) q[25];
U1q(0.712705875956659*pi,0.907725884475203*pi) q[26];
U1q(0.780680280913274*pi,1.867095434287597*pi) q[27];
U1q(0.351910328634266*pi,1.461216722717783*pi) q[28];
U1q(0.610434881375596*pi,0.622696402127493*pi) q[29];
U1q(0.0783095202905212*pi,1.787211986722114*pi) q[30];
U1q(0.732568302240906*pi,1.36453025297173*pi) q[31];
U1q(0.540295013580601*pi,0.286952341480355*pi) q[32];
U1q(0.730060846722476*pi,0.588375979848899*pi) q[33];
U1q(0.588821757712026*pi,0.560056916458882*pi) q[34];
U1q(0.664366671938338*pi,1.423900577405085*pi) q[35];
U1q(0.424335354002803*pi,0.180627001237607*pi) q[36];
U1q(0.474717306272423*pi,1.751668728959968*pi) q[37];
U1q(0.731223652949571*pi,0.0230914978979565*pi) q[38];
U1q(0.575484005311609*pi,1.315588954888415*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[1],q[7];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[18],q[19];
RZZ(0.5*pi) q[31],q[20];
RZZ(0.5*pi) q[21],q[30];
RZZ(0.5*pi) q[23],q[22];
RZZ(0.5*pi) q[32],q[28];
RZZ(0.5*pi) q[33],q[35];
RZZ(0.5*pi) q[39],q[38];
U1q(0.684276413099804*pi,0.61640899503591*pi) q[0];
U1q(0.533442894899508*pi,0.07530250055465992*pi) q[1];
U1q(0.480855803911926*pi,1.4413611676838598*pi) q[2];
U1q(0.482353753029836*pi,1.8391312404910298*pi) q[3];
U1q(0.716112704934709*pi,1.166239140854471*pi) q[4];
U1q(0.423038632990496*pi,0.22422043601523*pi) q[5];
U1q(0.595573635200144*pi,1.7338214946089598*pi) q[6];
U1q(0.702861373897034*pi,1.220822908512969*pi) q[7];
U1q(0.435854275020523*pi,1.7751505057308998*pi) q[8];
U1q(0.955949528155199*pi,1.4299221191689702*pi) q[9];
U1q(0.428074010775859*pi,0.06194683803226009*pi) q[10];
U1q(0.374905929453253*pi,0.07758367064664995*pi) q[11];
U1q(0.709803386326289*pi,0.95370791214679*pi) q[12];
U1q(0.759631425867268*pi,1.71678382193952*pi) q[13];
U1q(0.149782188824006*pi,1.8607150909472399*pi) q[14];
U1q(0.771979705164712*pi,1.69401188377573*pi) q[15];
U1q(0.820032567939024*pi,1.79557764458368*pi) q[16];
U1q(0.170103712373159*pi,0.21901467025163002*pi) q[17];
U1q(0.231270411798088*pi,0.75291020973562*pi) q[18];
U1q(0.205923819187715*pi,1.078295058335721*pi) q[19];
U1q(0.36704183335534*pi,0.16701337386600001*pi) q[20];
U1q(0.69962080845887*pi,0.17927892565735992*pi) q[21];
U1q(0.0927690339579226*pi,0.8611465204264901*pi) q[22];
U1q(0.650015977674694*pi,1.0214574196349862*pi) q[23];
U1q(0.194981161321077*pi,0.11986784162139008*pi) q[24];
U1q(0.749918176335462*pi,1.189822389948278*pi) q[25];
U1q(0.695531723489778*pi,0.256405189942772*pi) q[26];
U1q(0.386530053070589*pi,1.0083214060808499*pi) q[27];
U1q(0.444039934533183*pi,0.37732875829856005*pi) q[28];
U1q(0.593722901429877*pi,1.32506208944553*pi) q[29];
U1q(0.385340167028875*pi,0.7216119143894999*pi) q[30];
U1q(0.798197635716512*pi,1.3133668236736669*pi) q[31];
U1q(0.615144706955848*pi,0.32339603775844994*pi) q[32];
U1q(0.380150702939265*pi,0.70945906049381*pi) q[33];
U1q(0.183788589859663*pi,0.7943050172342798*pi) q[34];
U1q(0.0696019335225272*pi,1.6141364029642702*pi) q[35];
U1q(0.555654067822442*pi,0.15124547572824*pi) q[36];
U1q(0.458205140815196*pi,1.9038320520857104*pi) q[37];
U1q(0.202965773679792*pi,0.22524108841304002*pi) q[38];
U1q(0.358762309487599*pi,0.8210843949867299*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[27],q[7];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[19],q[9];
RZZ(0.5*pi) q[11],q[35];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[34],q[23];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[31],q[29];
RZZ(0.5*pi) q[32],q[39];
RZZ(0.5*pi) q[37],q[33];
U1q(0.503826646975721*pi,0.10896923586681018*pi) q[0];
U1q(0.731643352659863*pi,0.89117160354568*pi) q[1];
U1q(0.374733237946991*pi,1.93267848967991*pi) q[2];
U1q(0.896717174603661*pi,0.32423792249110983*pi) q[3];
U1q(0.734524921440229*pi,1.6272852416761499*pi) q[4];
U1q(0.562069381425295*pi,1.11714163471486*pi) q[5];
U1q(0.23041220223499*pi,0.09799939434538985*pi) q[6];
U1q(0.5110024281144*pi,0.5115694425024899*pi) q[7];
U1q(0.349133874913825*pi,0.2507236954751999*pi) q[8];
U1q(0.507573852215583*pi,0.5073605487304498*pi) q[9];
U1q(0.629489920506106*pi,1.0945476738398696*pi) q[10];
U1q(0.408766988398792*pi,1.1687942768860697*pi) q[11];
U1q(0.393683071069723*pi,0.25822524997765983*pi) q[12];
U1q(0.462184455898494*pi,1.0052670670015402*pi) q[13];
U1q(0.57007839624015*pi,0.9123603124568*pi) q[14];
U1q(0.372387149887064*pi,1.2990921952643104*pi) q[15];
U1q(0.769030563846378*pi,0.02490947058698012*pi) q[16];
U1q(0.592528376068626*pi,0.8071330834596799*pi) q[17];
U1q(0.427050288668603*pi,1.17660768147858*pi) q[18];
U1q(0.131724615275432*pi,0.9919972914338304*pi) q[19];
U1q(0.634196275475444*pi,1.2671656807537302*pi) q[20];
U1q(0.528792983705374*pi,1.3037964738094399*pi) q[21];
U1q(0.496223273785664*pi,0.5394169670278*pi) q[22];
U1q(0.482461869526827*pi,1.5400428398846402*pi) q[23];
U1q(0.557679014574202*pi,0.23281788876917986*pi) q[24];
U1q(0.683674233906359*pi,0.013453785497429926*pi) q[25];
U1q(0.38584444986926*pi,1.04907103437822*pi) q[26];
U1q(0.41570136137405*pi,1.7996876553913603*pi) q[27];
U1q(0.94068216182112*pi,1.10498705750015*pi) q[28];
U1q(0.396206386296014*pi,1.5031286480529502*pi) q[29];
U1q(0.747699180358158*pi,0.9235714767204799*pi) q[30];
U1q(0.914801925672511*pi,1.8986527401119*pi) q[31];
U1q(0.00926982131128561*pi,0.18221734503849008*pi) q[32];
U1q(0.762930730475715*pi,1.67291273541184*pi) q[33];
U1q(0.303577634672444*pi,1.0300688158277396*pi) q[34];
U1q(0.805089971065374*pi,1.8569337078011596*pi) q[35];
U1q(0.711518061020384*pi,1.67836510195517*pi) q[36];
U1q(0.231755117615203*pi,0.2750588285729103*pi) q[37];
U1q(0.54867870429999*pi,1.3122264959716503*pi) q[38];
U1q(0.467370582292232*pi,0.8201678417841096*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[12],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[38],q[7];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[21],q[14];
RZZ(0.5*pi) q[15],q[24];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[31],q[30];
U1q(0.258380723795486*pi,1.7861992637344004*pi) q[0];
U1q(0.365259358850048*pi,1.1463496774087796*pi) q[1];
U1q(0.265435383253266*pi,1.68943609017626*pi) q[2];
U1q(0.756874895131452*pi,1.6412271709942603*pi) q[3];
U1q(0.699743845654203*pi,0.6431757596669199*pi) q[4];
U1q(0.486813562390483*pi,0.17255663849867986*pi) q[5];
U1q(0.417241591906127*pi,1.10479645356582*pi) q[6];
U1q(0.562620763894449*pi,0.6691100742072997*pi) q[7];
U1q(0.311438605682105*pi,0.7728358835004201*pi) q[8];
U1q(0.517209020179652*pi,1.7695552460152904*pi) q[9];
U1q(0.52301318423139*pi,1.9654981239077394*pi) q[10];
U1q(0.500557437758178*pi,0.6263004961480698*pi) q[11];
U1q(0.152499771295944*pi,0.6041933611225598*pi) q[12];
U1q(0.4603089797702*pi,0.54185859228014*pi) q[13];
U1q(0.427554545640301*pi,1.4552304716096298*pi) q[14];
U1q(0.611135846289737*pi,0.8527329157129797*pi) q[15];
U1q(0.789558051072039*pi,0.5338925366159399*pi) q[16];
U1q(0.358374625748738*pi,1.1526934438317404*pi) q[17];
U1q(0.963241462234498*pi,1.4129309987820298*pi) q[18];
U1q(0.194843967824675*pi,1.0995022351026096*pi) q[19];
U1q(0.785357459072232*pi,0.9272120347522605*pi) q[20];
U1q(0.382286212145939*pi,0.3452596884210104*pi) q[21];
U1q(0.387399817408894*pi,1.7526699677359003*pi) q[22];
U1q(0.65310625039105*pi,1.8405336765543598*pi) q[23];
U1q(0.293595143798095*pi,1.9957301606749498*pi) q[24];
U1q(0.449953861521873*pi,1.9529875502000698*pi) q[25];
U1q(0.502660389827751*pi,1.6050042917007898*pi) q[26];
U1q(0.403309427813302*pi,1.7217122442130997*pi) q[27];
U1q(0.619141772426369*pi,0.8343418011691996*pi) q[28];
U1q(0.731122241880599*pi,1.0719973619107597*pi) q[29];
U1q(0.260075333436882*pi,1.78642985454597*pi) q[30];
U1q(0.367631034300741*pi,0.7855764651933503*pi) q[31];
U1q(0.301312607367012*pi,0.6804517890675399*pi) q[32];
U1q(0.896820209415904*pi,1.70109882571652*pi) q[33];
U1q(0.541287357025994*pi,0.5796639441041798*pi) q[34];
U1q(0.359461927311043*pi,1.8172029155527802*pi) q[35];
U1q(0.640893983230121*pi,1.3963014657139001*pi) q[36];
U1q(0.604474257084839*pi,1.1817072768329897*pi) q[37];
U1q(0.64875384802293*pi,0.05207498914653996*pi) q[38];
U1q(0.329427983198032*pi,0.17050141848185962*pi) q[39];
RZZ(0.5*pi) q[1],q[0];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[34],q[9];
RZZ(0.5*pi) q[10],q[35];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[29],q[20];
RZZ(0.5*pi) q[21],q[39];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[37],q[27];
RZZ(0.5*pi) q[31],q[28];
U1q(0.238662664233015*pi,0.44628752940607974*pi) q[0];
U1q(0.092888738866014*pi,1.5829023105181204*pi) q[1];
U1q(0.453375186480233*pi,0.36105430047028086*pi) q[2];
U1q(0.232530501902092*pi,1.4961349036833997*pi) q[3];
U1q(0.472299866812869*pi,1.5244322886638004*pi) q[4];
U1q(0.465055123945538*pi,0.014235065136560365*pi) q[5];
U1q(0.692530214931696*pi,0.3078475133932006*pi) q[6];
U1q(0.64927045405822*pi,1.9808776757495403*pi) q[7];
U1q(0.777461741066923*pi,1.3278852232140004*pi) q[8];
U1q(0.440311991459026*pi,0.6673017343599401*pi) q[9];
U1q(0.393818965952727*pi,0.2086793891485108*pi) q[10];
U1q(0.50462705225652*pi,0.2054971517373403*pi) q[11];
U1q(0.556137885629413*pi,0.2989807222858598*pi) q[12];
U1q(0.875433564205617*pi,1.3483639500305902*pi) q[13];
U1q(0.514274427013123*pi,1.4546859094287798*pi) q[14];
U1q(0.37515615736209*pi,1.2286125988388008*pi) q[15];
U1q(0.832751814799066*pi,0.9521393105067402*pi) q[16];
U1q(0.552828631376928*pi,1.5263254487008595*pi) q[17];
U1q(0.384625269338247*pi,1.3788578651254007*pi) q[18];
U1q(0.773264057166744*pi,1.4698065541611207*pi) q[19];
U1q(0.0935258747232987*pi,0.34100988666980037*pi) q[20];
U1q(0.626382456317178*pi,1.5284915525440699*pi) q[21];
U1q(0.0926692045851719*pi,1.5785761522016903*pi) q[22];
U1q(0.59563994610493*pi,1.5597742098755196*pi) q[23];
U1q(0.470556577197724*pi,1.1491232820982704*pi) q[24];
U1q(0.488189658238282*pi,1.6797357108086803*pi) q[25];
U1q(0.36322803008188*pi,0.39767887680432956*pi) q[26];
U1q(0.887236517856199*pi,0.21279991797002928*pi) q[27];
U1q(0.715893787490043*pi,0.29373326547685075*pi) q[28];
U1q(0.283313114559594*pi,1.5012286635951*pi) q[29];
U1q(0.781307499372544*pi,1.4876235051847004*pi) q[30];
U1q(0.0712350652245838*pi,0.9623198086422295*pi) q[31];
U1q(0.591745729273228*pi,0.7486796031840299*pi) q[32];
U1q(0.73873849768633*pi,0.44672687594047034*pi) q[33];
U1q(0.13156178007841*pi,0.4526528127789806*pi) q[34];
U1q(0.465336652928894*pi,1.5405943403969395*pi) q[35];
U1q(0.374913866003629*pi,1.5400860765691702*pi) q[36];
U1q(0.312449628472429*pi,0.9944470621917993*pi) q[37];
U1q(0.870211889474201*pi,1.2993271118004799*pi) q[38];
U1q(0.440781375887785*pi,0.7998775797003592*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[6],q[2];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[10],q[14];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[12],q[28];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[30],q[33];
RZZ(0.5*pi) q[39],q[35];
RZZ(0.5*pi) q[37],q[38];
U1q(0.524178290183601*pi,1.3900778998637993*pi) q[0];
U1q(0.673950343336802*pi,0.7404022611658299*pi) q[1];
U1q(0.603899217202241*pi,1.0644640812977002*pi) q[2];
U1q(0.366184394491161*pi,1.4569648139114992*pi) q[3];
U1q(0.745948393985345*pi,1.60962592885266*pi) q[4];
U1q(0.634844022773583*pi,0.7282878201441001*pi) q[5];
U1q(0.21398835152464*pi,0.3206784858915004*pi) q[6];
U1q(0.736646796989026*pi,1.5053796852475703*pi) q[7];
U1q(0.760879052038899*pi,0.5674490651399005*pi) q[8];
U1q(0.851688526791271*pi,1.4178720626724992*pi) q[9];
U1q(0.143314848149572*pi,1.5739126795001006*pi) q[10];
U1q(0.168359568749102*pi,1.4893755866998006*pi) q[11];
U1q(0.496655264040808*pi,0.9018216218522994*pi) q[12];
U1q(0.581123848800337*pi,0.4105098824324198*pi) q[13];
U1q(0.577090824029054*pi,1.5664254177450907*pi) q[14];
U1q(0.320549598418399*pi,0.022344523380599668*pi) q[15];
U1q(0.595678168311989*pi,0.08700398017156985*pi) q[16];
U1q(0.647691255235336*pi,0.5625157486757004*pi) q[17];
U1q(0.736728182436552*pi,1.6141890076292995*pi) q[18];
U1q(0.199517734766122*pi,0.4575566011004*pi) q[19];
U1q(0.485088746106167*pi,0.25283994831410084*pi) q[20];
U1q(0.335168417828712*pi,1.3892743604955005*pi) q[21];
U1q(0.531166887999424*pi,1.2442395468372602*pi) q[22];
U1q(0.707980984793947*pi,0.18569929749100922*pi) q[23];
U1q(0.188503795308652*pi,0.9591219103944297*pi) q[24];
U1q(0.366426406367532*pi,0.14294392934984046*pi) q[25];
U1q(0.906787033762345*pi,0.8411550489188802*pi) q[26];
U1q(0.302266618648856*pi,1.6104611813898*pi) q[27];
U1q(0.465966799532535*pi,1.9160684699390007*pi) q[28];
U1q(0.721291902398771*pi,0.04497190538089946*pi) q[29];
U1q(0.474628652512927*pi,0.48992991116739937*pi) q[30];
U1q(0.152913807490812*pi,0.06573135657800044*pi) q[31];
U1q(0.483501039830852*pi,1.8217224093501994*pi) q[32];
U1q(0.975992754541532*pi,1.3221711392321502*pi) q[33];
U1q(0.257275006864248*pi,1.6872914047586995*pi) q[34];
U1q(0.78843423076646*pi,0.2872008683043994*pi) q[35];
U1q(0.500191642727619*pi,1.8639984244055396*pi) q[36];
U1q(0.734275436822938*pi,1.5798521680607003*pi) q[37];
U1q(0.20634871881408*pi,1.0855268394544009*pi) q[38];
U1q(0.890628477653105*pi,1.3497958975223003*pi) q[39];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[13],q[2];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[4],q[39];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[6],q[16];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[27],q[9];
RZZ(0.5*pi) q[12],q[37];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[25],q[35];
RZZ(0.5*pi) q[26],q[29];
RZZ(0.5*pi) q[31],q[32];
RZZ(0.5*pi) q[34],q[33];
U1q(0.719319187037571*pi,0.44125836678080077*pi) q[0];
U1q(0.110315495008487*pi,0.9197370840372994*pi) q[1];
U1q(0.578343014023783*pi,0.5685911991082993*pi) q[2];
U1q(0.455036505636888*pi,0.7573981385347999*pi) q[3];
U1q(0.370826279477418*pi,0.7772612701252406*pi) q[4];
U1q(0.433247129423774*pi,1.9105047072975*pi) q[5];
U1q(0.692544007670035*pi,1.3345701106022005*pi) q[6];
U1q(0.663103908730816*pi,0.2767235475642291*pi) q[7];
U1q(0.724299984955752*pi,0.8240866114380001*pi) q[8];
U1q(0.584188026608751*pi,0.2782067565819002*pi) q[9];
U1q(0.466252566121432*pi,0.5341131752345998*pi) q[10];
U1q(0.522715156372596*pi,1.2428742456117998*pi) q[11];
U1q(0.338295816904422*pi,1.9997363504147003*pi) q[12];
U1q(0.272309426907743*pi,1.8149648774053997*pi) q[13];
U1q(0.729580559952441*pi,0.32403765838239984*pi) q[14];
U1q(0.597160468834815*pi,0.3521454221195004*pi) q[15];
U1q(0.797895441050155*pi,1.2978236788850008*pi) q[16];
U1q(0.409677453299007*pi,0.009831576238200412*pi) q[17];
U1q(0.618672552801606*pi,1.5653014762149002*pi) q[18];
U1q(0.675047805177113*pi,0.5746135785239002*pi) q[19];
U1q(0.150906445355467*pi,1.1544731667801003*pi) q[20];
U1q(0.432778808042229*pi,0.8563848003550998*pi) q[21];
U1q(0.425386187471973*pi,1.2924212499414995*pi) q[22];
U1q(0.754245392660101*pi,0.9298911672904993*pi) q[23];
U1q(0.192149092905798*pi,0.42482627299139963*pi) q[24];
U1q(0.242025654371019*pi,1.5433880702276994*pi) q[25];
U1q(0.346404539153621*pi,0.3627121872777792*pi) q[26];
U1q(0.130277037396333*pi,0.9633265142234002*pi) q[27];
U1q(0.539174727256029*pi,1.2785301320731008*pi) q[28];
U1q(0.600565654253155*pi,0.3507158064162006*pi) q[29];
U1q(0.516245089172419*pi,0.5967193928116004*pi) q[30];
U1q(0.431972202948091*pi,1.2283851962616996*pi) q[31];
U1q(0.473098976515161*pi,1.4483963440330996*pi) q[32];
U1q(0.679089048526179*pi,1.07746590167784*pi) q[33];
U1q(0.714988238740656*pi,1.1120379179949005*pi) q[34];
U1q(0.398190302085998*pi,1.3179018906576*pi) q[35];
U1q(0.649623782058285*pi,1.0576697650904796*pi) q[36];
U1q(0.347958291338696*pi,0.2549870084318009*pi) q[37];
U1q(0.347662279278731*pi,1.4070523810267996*pi) q[38];
U1q(0.413523973771216*pi,1.3431129214952993*pi) q[39];
RZZ(0.5*pi) q[0],q[11];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[34],q[3];
RZZ(0.5*pi) q[14],q[4];
RZZ(0.5*pi) q[31],q[5];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[9],q[7];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[25],q[12];
RZZ(0.5*pi) q[13],q[22];
RZZ(0.5*pi) q[15],q[35];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[21],q[24];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[32],q[29];
RZZ(0.5*pi) q[39],q[30];
U1q(0.0535532162289165*pi,1.4494933355877997*pi) q[0];
U1q(0.522601452159794*pi,1.6183492990031993*pi) q[1];
U1q(0.766243816929341*pi,1.2603567916399*pi) q[2];
U1q(0.14647573374476*pi,1.6054829881120014*pi) q[3];
U1q(0.470945178577904*pi,0.6609306672991995*pi) q[4];
U1q(0.285960952745351*pi,1.197250423645201*pi) q[5];
U1q(0.341355254164469*pi,1.2454596855548985*pi) q[6];
U1q(0.40766146498155*pi,1.3243718263590996*pi) q[7];
U1q(0.502524808716972*pi,1.5057443524905985*pi) q[8];
U1q(0.778690858910603*pi,1.1651938102044994*pi) q[9];
U1q(0.707694665525362*pi,0.8567637918007005*pi) q[10];
U1q(0.236874058403609*pi,0.6041775650431003*pi) q[11];
U1q(0.470139347330329*pi,1.6818578620598998*pi) q[12];
U1q(0.464182543553856*pi,1.4262101216025993*pi) q[13];
U1q(0.410553678699484*pi,1.5472109320840008*pi) q[14];
U1q(0.640433311846794*pi,0.37243262479480066*pi) q[15];
U1q(0.699861283147184*pi,0.8125032199661*pi) q[16];
U1q(0.248006942286873*pi,1.8545872452598005*pi) q[17];
U1q(0.717210696784639*pi,1.4180095022625991*pi) q[18];
U1q(0.135885534107396*pi,0.47368602920399994*pi) q[19];
U1q(0.448972404464166*pi,1.2010496776540016*pi) q[20];
U1q(0.729235525600839*pi,0.8817359768180992*pi) q[21];
U1q(0.775315275381961*pi,0.7925553100967999*pi) q[22];
U1q(0.630110515999346*pi,1.4740986074136995*pi) q[23];
U1q(0.7094106624851*pi,1.3977823625046994*pi) q[24];
U1q(0.401863712806775*pi,0.030376743588199417*pi) q[25];
U1q(0.637216884268475*pi,0.12644068967249922*pi) q[26];
U1q(0.346300736504618*pi,0.005278271988998995*pi) q[27];
U1q(0.337407047847459*pi,0.9970362581478014*pi) q[28];
U1q(0.489939602954271*pi,1.1270309969174015*pi) q[29];
U1q(0.838545319340492*pi,0.5042643804214002*pi) q[30];
U1q(0.691029451874881*pi,1.0519585374231006*pi) q[31];
U1q(0.259547602402372*pi,0.7990020019054*pi) q[32];
U1q(0.635619918581681*pi,1.3360160531145997*pi) q[33];
U1q(0.551610161177998*pi,0.3444694941856987*pi) q[34];
U1q(0.374773699823262*pi,0.2973489729985985*pi) q[35];
U1q(0.45122492837301*pi,0.7909966039451799*pi) q[36];
U1q(0.362791496042879*pi,0.1610748299172009*pi) q[37];
U1q(0.351081141222126*pi,1.7940711853395008*pi) q[38];
U1q(0.548118313741024*pi,0.007622351552399209*pi) q[39];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[21],q[8];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[15],q[16];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[20],q[37];
RZZ(0.5*pi) q[36],q[22];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[33],q[27];
RZZ(0.5*pi) q[34],q[30];
U1q(0.862440420579386*pi,1.746703084714401*pi) q[0];
U1q(0.445226851768581*pi,1.5712063525526005*pi) q[1];
U1q(0.113630962779187*pi,1.1987885604356983*pi) q[2];
U1q(0.403747697323091*pi,0.5123069836451997*pi) q[3];
U1q(0.508106552639334*pi,1.8502172940135004*pi) q[4];
U1q(0.625131630865551*pi,0.30305612873589993*pi) q[5];
U1q(0.689902654453525*pi,1.3172513368184013*pi) q[6];
U1q(0.465187081959424*pi,0.5137652212847001*pi) q[7];
U1q(0.634074262673037*pi,1.6347532888928988*pi) q[8];
U1q(0.798103335078179*pi,1.4090687869304013*pi) q[9];
U1q(0.462974743179303*pi,0.5911912679993989*pi) q[10];
U1q(0.0637902008234066*pi,0.5631300616621004*pi) q[11];
U1q(0.743306379392266*pi,0.14742561027570034*pi) q[12];
U1q(0.420598463225354*pi,0.4740490296552*pi) q[13];
U1q(0.370706539723773*pi,1.2926307391615008*pi) q[14];
U1q(0.728421592620311*pi,1.3851253281047988*pi) q[15];
U1q(0.60280984282009*pi,1.3635066097291002*pi) q[16];
U1q(0.323288568814608*pi,1.5878587249067984*pi) q[17];
U1q(0.803974437269541*pi,1.7602362932321007*pi) q[18];
U1q(0.582391877982221*pi,1.8579101986919007*pi) q[19];
U1q(0.827031546808623*pi,1.8179687760062002*pi) q[20];
U1q(0.501563526468404*pi,0.1661927445535003*pi) q[21];
U1q(0.493671963556841*pi,1.9833310532032016*pi) q[22];
U1q(0.735805988348991*pi,0.6519955956319006*pi) q[23];
U1q(0.869216012872198*pi,0.5233454884535007*pi) q[24];
U1q(0.280468469992425*pi,0.2915598220524984*pi) q[25];
U1q(0.409426980504141*pi,0.23845510021459937*pi) q[26];
U1q(0.351683376357137*pi,1.4208064058007999*pi) q[27];
U1q(0.315418326411161*pi,1.2822703353436005*pi) q[28];
U1q(0.479745111230968*pi,1.4492284437164002*pi) q[29];
U1q(0.388831729387946*pi,0.3902944960588002*pi) q[30];
U1q(0.214620261607709*pi,0.09586292393819917*pi) q[31];
U1q(0.366120122531984*pi,1.0560432438341003*pi) q[32];
U1q(0.10723061462853*pi,0.8234973272420003*pi) q[33];
U1q(0.650563511580626*pi,1.3449994835848003*pi) q[34];
U1q(0.710322055853577*pi,1.3539391747086*pi) q[35];
U1q(0.813142743062908*pi,1.2641691649317295*pi) q[36];
U1q(0.780230239588742*pi,0.7769036472537998*pi) q[37];
U1q(0.36784513056077*pi,0.12255037953010017*pi) q[38];
U1q(0.692811464305683*pi,0.5538707065682011*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[30];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[8];
RZZ(0.5*pi) q[32],q[6];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[36];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[27],q[22];
RZZ(0.5*pi) q[29],q[28];
U1q(0.286182584033949*pi,1.8697593487912982*pi) q[0];
U1q(0.275522877950377*pi,1.9790720547791985*pi) q[1];
U1q(0.260129486111499*pi,1.9917721815104983*pi) q[2];
U1q(0.635460212257335*pi,1.2600170034242986*pi) q[3];
U1q(0.647651103357949*pi,0.7874445948882993*pi) q[4];
U1q(0.601722827125486*pi,0.03789692677879941*pi) q[5];
U1q(0.692188969434791*pi,1.8315734956299998*pi) q[6];
U1q(0.480681020296817*pi,0.20097065852720064*pi) q[7];
U1q(0.503540721938585*pi,1.194161741579201*pi) q[8];
U1q(0.291178300193243*pi,0.4941671789736013*pi) q[9];
U1q(0.251572512003726*pi,0.6501306099537985*pi) q[10];
U1q(0.761246655652942*pi,1.3526288593578997*pi) q[11];
U1q(0.578271876739913*pi,1.1020814892345001*pi) q[12];
U1q(0.441744066969259*pi,0.113642635467599*pi) q[13];
U1q(0.262617076550076*pi,1.8481036588706985*pi) q[14];
U1q(0.368374554676281*pi,1.1195723098241004*pi) q[15];
U1q(0.543232820185882*pi,1.8849709676996014*pi) q[16];
U1q(0.368036929450306*pi,0.3482854197629983*pi) q[17];
U1q(0.372600844067014*pi,0.5585323364779988*pi) q[18];
U1q(0.705432101583513*pi,0.2806706234155989*pi) q[19];
U1q(0.602428334857288*pi,0.37099828313050054*pi) q[20];
U1q(0.832627011154789*pi,0.4475780709947017*pi) q[21];
U1q(0.131264831339447*pi,0.21737812180749927*pi) q[22];
U1q(0.243570055435816*pi,0.13483490359000072*pi) q[23];
U1q(0.472702877590196*pi,0.9310328170681998*pi) q[24];
U1q(0.755956412878602*pi,1.719829628192901*pi) q[25];
U1q(0.151590746184697*pi,0.04608065797060057*pi) q[26];
U1q(0.446737070447108*pi,0.7069807421417984*pi) q[27];
U1q(0.186130694383291*pi,1.5931379183695995*pi) q[28];
U1q(0.262713112254759*pi,0.6849105358640983*pi) q[29];
U1q(0.379292242905873*pi,0.24771047897279885*pi) q[30];
U1q(0.666852539711084*pi,0.5140136378782003*pi) q[31];
U1q(0.491096892509359*pi,0.8367850348317987*pi) q[32];
U1q(0.257300987107291*pi,0.5716257673832992*pi) q[33];
U1q(0.323506312233017*pi,1.282111794647399*pi) q[34];
U1q(0.201905986890851*pi,0.9718813633792998*pi) q[35];
U1q(0.741644291886532*pi,0.7916189564343092*pi) q[36];
U1q(0.822401218509675*pi,0.10043938834819954*pi) q[37];
U1q(0.147911374081231*pi,1.795720217074301*pi) q[38];
U1q(0.463556532628121*pi,1.2045938611863996*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[1],q[17];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[13];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[16],q[7];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[14],q[35];
RZZ(0.5*pi) q[34],q[18];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[25],q[22];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[29],q[30];
RZZ(0.5*pi) q[32],q[36];
RZZ(0.5*pi) q[33],q[38];
RZZ(0.5*pi) q[37],q[39];
U1q(0.431120170295816*pi,0.23037148112619832*pi) q[0];
U1q(0.327861432551762*pi,0.6301124663156017*pi) q[1];
U1q(0.415796471561599*pi,0.8213694306963006*pi) q[2];
U1q(0.318236566616513*pi,1.699491101678401*pi) q[3];
U1q(0.848116965426432*pi,1.9903727295343003*pi) q[4];
U1q(0.437802698978437*pi,0.5629289423870993*pi) q[5];
U1q(0.513221121953212*pi,1.3578682877055996*pi) q[6];
U1q(0.493428093089122*pi,0.576089236642499*pi) q[7];
U1q(0.529310265664158*pi,1.937584452621099*pi) q[8];
U1q(0.475853739346733*pi,0.7500951952577992*pi) q[9];
U1q(0.469038546856992*pi,0.09417103740640087*pi) q[10];
U1q(0.638650159827087*pi,1.2818451068010006*pi) q[11];
U1q(0.0935821894820134*pi,0.1332756514624016*pi) q[12];
U1q(0.497694144118321*pi,1.1232999536154011*pi) q[13];
U1q(0.67010184475628*pi,0.9217606505450995*pi) q[14];
U1q(0.328370388856457*pi,0.7997518750889014*pi) q[15];
U1q(0.716081013941775*pi,1.3777443875922017*pi) q[16];
U1q(0.249366000155451*pi,0.1654642678387006*pi) q[17];
U1q(0.266448644108334*pi,0.2195681050685998*pi) q[18];
U1q(0.17496031340262*pi,0.8245834860507983*pi) q[19];
U1q(0.669410461334385*pi,1.8258425171284998*pi) q[20];
U1q(0.402094972280181*pi,0.5367947529780999*pi) q[21];
U1q(0.802222346170544*pi,0.15906830758660107*pi) q[22];
U1q(0.419727091878213*pi,0.5217591799376997*pi) q[23];
U1q(0.583333613554729*pi,0.3959468510554984*pi) q[24];
U1q(0.530606550071624*pi,1.9673106614026992*pi) q[25];
U1q(0.686817892555752*pi,0.9389900162663984*pi) q[26];
U1q(0.357195279238693*pi,1.4457328207986002*pi) q[27];
U1q(0.0609410154741925*pi,0.38160015742300146*pi) q[28];
U1q(0.700205415145879*pi,1.4563789596094985*pi) q[29];
U1q(0.556810828723396*pi,1.3784649619707992*pi) q[30];
U1q(0.77664836590725*pi,1.2657988898649002*pi) q[31];
U1q(0.350921103590523*pi,1.4686685593222002*pi) q[32];
U1q(0.411866449672642*pi,1.6717444401129988*pi) q[33];
U1q(0.343988740790163*pi,0.6259138637518014*pi) q[34];
U1q(0.776211798104179*pi,1.2167135166503016*pi) q[35];
U1q(0.565075995216614*pi,1.0729815825469995*pi) q[36];
U1q(0.704942354920406*pi,0.6142005247405997*pi) q[37];
U1q(0.187560282040677*pi,1.4252833046428002*pi) q[38];
U1q(0.539479320799243*pi,1.9683996186221009*pi) q[39];
rz(2.2894712931149996*pi) q[0];
rz(0.4316182379997997*pi) q[1];
rz(2.7747810090396996*pi) q[2];
rz(1.7078141550049004*pi) q[3];
rz(1.9271400019124982*pi) q[4];
rz(2.7065551435687993*pi) q[5];
rz(2.2028238062328995*pi) q[6];
rz(0.5619389224873999*pi) q[7];
rz(0.915017357082899*pi) q[8];
rz(0.6496206993620994*pi) q[9];
rz(2.5904409496095013*pi) q[10];
rz(1.5114510717565004*pi) q[11];
rz(3.7551277191262002*pi) q[12];
rz(2.3096301909248*pi) q[13];
rz(3.1533182880831987*pi) q[14];
rz(3.9681881097544007*pi) q[15];
rz(1.7462384697250997*pi) q[16];
rz(2.823496400836799*pi) q[17];
rz(0.028994530349599756*pi) q[18];
rz(0.8055929466684013*pi) q[19];
rz(3.2512701632856*pi) q[20];
rz(3.5898748075507996*pi) q[21];
rz(3.577163581090801*pi) q[22];
rz(1.7798037473827009*pi) q[23];
rz(3.4093046772165003*pi) q[24];
rz(2.7820190057563003*pi) q[25];
rz(1.7026398699669016*pi) q[26];
rz(3.3842878059671015*pi) q[27];
rz(0.438942915966301*pi) q[28];
rz(0.14744031307829886*pi) q[29];
rz(1.7165151211220007*pi) q[30];
rz(0.41491568829500025*pi) q[31];
rz(3.2265216347217986*pi) q[32];
rz(1.0478424135143989*pi) q[33];
rz(3.1832546422622983*pi) q[34];
rz(2.415029339824901*pi) q[35];
rz(3.0045369729889995*pi) q[36];
rz(3.3208963681057995*pi) q[37];
rz(1.1257236272427988*pi) q[38];
rz(1.2140413091864986*pi) q[39];
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
