OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.132878711099741*pi,0.0345144620647231*pi) q[0];
U1q(0.359212379453448*pi,1.346466953915743*pi) q[1];
U1q(0.330314052520933*pi,1.305960436767847*pi) q[2];
U1q(1.36161423976044*pi,0.554348014607271*pi) q[3];
U1q(1.51004624585604*pi,0.20039002803575068*pi) q[4];
U1q(1.44143749049547*pi,1.5348329582210773*pi) q[5];
U1q(1.84819830605081*pi,0.23492511205679528*pi) q[6];
U1q(1.57989940437503*pi,0.33756147348570614*pi) q[7];
U1q(1.42629108872255*pi,1.7607976647126613*pi) q[8];
U1q(1.31104486165108*pi,1.5680324233587268*pi) q[9];
U1q(1.72449763946283*pi,0.3458552580207785*pi) q[10];
U1q(3.490085517157544*pi,0.8008232975572278*pi) q[11];
U1q(0.756371682599181*pi,0.596057136893034*pi) q[12];
U1q(0.382880021667634*pi,1.9885561040030524*pi) q[13];
U1q(1.49782356286289*pi,0.8958012851692496*pi) q[14];
U1q(0.455119909929831*pi,0.596444551255565*pi) q[15];
U1q(1.29275156863519*pi,1.903510606538116*pi) q[16];
U1q(1.46814794233177*pi,1.7291964541196947*pi) q[17];
U1q(0.860992421227862*pi,0.361063556804177*pi) q[18];
U1q(0.727892068914786*pi,1.010053648537335*pi) q[19];
U1q(0.900299959064042*pi,1.892373330253384*pi) q[20];
U1q(0.406616970749166*pi,0.143186414663803*pi) q[21];
U1q(1.67824190693798*pi,1.0896428198055226*pi) q[22];
U1q(1.41566698722016*pi,0.19788572702153356*pi) q[23];
U1q(1.18886617777487*pi,1.8265403297157585*pi) q[24];
U1q(0.594630062875135*pi,0.68553945194704*pi) q[25];
U1q(3.506560354735796*pi,0.4899917479009798*pi) q[26];
U1q(1.50132998464332*pi,1.6288846614401224*pi) q[27];
U1q(1.35667461925444*pi,0.37076569218924926*pi) q[28];
U1q(1.69194227477596*pi,1.9699353583365062*pi) q[29];
U1q(3.33653438622163*pi,1.0451325857457734*pi) q[30];
U1q(0.430040705114152*pi,0.854149124320109*pi) q[31];
U1q(1.42946966305784*pi,1.4076800709797643*pi) q[32];
U1q(0.362946390563548*pi,1.749810899577478*pi) q[33];
U1q(0.63610615043126*pi,1.518872336801099*pi) q[34];
U1q(1.93720304199315*pi,0.5444580883243498*pi) q[35];
U1q(0.42953286292293*pi,1.466435131835222*pi) q[36];
U1q(0.73217305303267*pi,0.638237007998517*pi) q[37];
U1q(3.255356370400301*pi,1.431922269600418*pi) q[38];
U1q(0.460754951705264*pi,0.0339019147678112*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[8],q[15];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[28],q[19];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[31],q[39];
RZZ(0.5*pi) q[35],q[38];
U1q(0.859620283669499*pi,0.45627667606874*pi) q[0];
U1q(0.665931957032461*pi,0.2227492457912701*pi) q[1];
U1q(0.320681752168941*pi,0.14932499121624998*pi) q[2];
U1q(0.471307809183715*pi,0.6112534980221309*pi) q[3];
U1q(0.34752239642197*pi,0.7463970299495606*pi) q[4];
U1q(0.593483072391106*pi,0.27272813231572757*pi) q[5];
U1q(0.783775645865476*pi,1.6511554333963554*pi) q[6];
U1q(0.332065162672106*pi,0.312911642622546*pi) q[7];
U1q(0.212762499359479*pi,0.2304366054409812*pi) q[8];
U1q(0.369941444850936*pi,1.6973127005066564*pi) q[9];
U1q(0.482903217259384*pi,0.02063974225835863*pi) q[10];
U1q(0.705335715546129*pi,0.5263846567871229*pi) q[11];
U1q(0.646787224304459*pi,0.83447148793679*pi) q[12];
U1q(0.459433058676634*pi,0.8675024777078499*pi) q[13];
U1q(0.872829899235206*pi,0.8018397301652898*pi) q[14];
U1q(0.697712335110492*pi,0.37726901403319*pi) q[15];
U1q(0.925835234194168*pi,0.6534069087117862*pi) q[16];
U1q(0.502792964664056*pi,1.0610665232348548*pi) q[17];
U1q(0.902140826049996*pi,1.5178007162168101*pi) q[18];
U1q(0.454869979208406*pi,0.8991014229073002*pi) q[19];
U1q(0.312580474357863*pi,0.5644214750299699*pi) q[20];
U1q(0.239226019592009*pi,0.5897968804535898*pi) q[21];
U1q(0.798202299138304*pi,1.2816704401727526*pi) q[22];
U1q(0.238377137937571*pi,1.3510319112844535*pi) q[23];
U1q(0.724414118748761*pi,0.3688942824693582*pi) q[24];
U1q(0.471829028411107*pi,0.90455663900325*pi) q[25];
U1q(0.476590134419528*pi,0.23919731361694008*pi) q[26];
U1q(0.338630303835682*pi,1.1315308301917524*pi) q[27];
U1q(0.237123931863681*pi,0.11826771885549925*pi) q[28];
U1q(0.192867256851777*pi,1.4745801171892463*pi) q[29];
U1q(0.752452390551622*pi,0.3288418840503575*pi) q[30];
U1q(0.374876616676999*pi,0.49476445567244*pi) q[31];
U1q(0.10045179499242*pi,1.4819270992583942*pi) q[32];
U1q(0.782639301813296*pi,0.14257658396139994*pi) q[33];
U1q(0.180252335136177*pi,0.86354090907792*pi) q[34];
U1q(0.340024875349038*pi,1.1588492046166499*pi) q[35];
U1q(0.653712212755041*pi,1.9297146539307501*pi) q[36];
U1q(0.552831265640836*pi,0.155027295717641*pi) q[37];
U1q(0.272568555917096*pi,0.835081342127288*pi) q[38];
U1q(0.224827533431626*pi,0.05452676540902002*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[1],q[4];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[38],q[10];
RZZ(0.5*pi) q[14],q[11];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[35],q[39];
U1q(0.188084487236584*pi,1.1750278200614899*pi) q[0];
U1q(0.124871135822001*pi,1.78395310151687*pi) q[1];
U1q(0.630735839646314*pi,1.8682430796523297*pi) q[2];
U1q(0.623573510573543*pi,0.08529632804706111*pi) q[3];
U1q(0.952636681827652*pi,1.0427673048702815*pi) q[4];
U1q(0.562590829516081*pi,0.8452429819409977*pi) q[5];
U1q(0.559285007094763*pi,1.7153777577031954*pi) q[6];
U1q(0.513175477215425*pi,0.233551565324706*pi) q[7];
U1q(0.248598703192695*pi,1.5448580852303513*pi) q[8];
U1q(0.451425641012476*pi,1.5833407412903364*pi) q[9];
U1q(0.29386520374557*pi,1.2817594747366883*pi) q[10];
U1q(0.735732539329757*pi,0.8312046344091077*pi) q[11];
U1q(0.537007757414406*pi,0.9968780469776002*pi) q[12];
U1q(0.816845922910767*pi,1.8983238595762701*pi) q[13];
U1q(0.504536769979797*pi,1.2836426535114898*pi) q[14];
U1q(0.765611104634572*pi,1.80988099390383*pi) q[15];
U1q(0.470182940250804*pi,0.6941135907951255*pi) q[16];
U1q(0.575553039809543*pi,1.946110518720185*pi) q[17];
U1q(0.341904937113912*pi,1.59217087033789*pi) q[18];
U1q(0.434046568255889*pi,1.2811924235166199*pi) q[19];
U1q(0.506955232347952*pi,1.6920088246124196*pi) q[20];
U1q(0.900662085539139*pi,0.9280688423060299*pi) q[21];
U1q(0.702379324046984*pi,1.1074994642204725*pi) q[22];
U1q(0.800445181133743*pi,0.4888524946235435*pi) q[23];
U1q(0.538720793194352*pi,1.9186793796763082*pi) q[24];
U1q(0.480305674984285*pi,1.2359409507400096*pi) q[25];
U1q(0.751828394581896*pi,0.8878475297909199*pi) q[26];
U1q(0.343219998989344*pi,1.5850462409081425*pi) q[27];
U1q(0.770671362860269*pi,1.0532768544665192*pi) q[28];
U1q(0.213997217224906*pi,0.9848941544517356*pi) q[29];
U1q(0.294918688789138*pi,1.0510962127607533*pi) q[30];
U1q(0.908013629360044*pi,1.9491027877736*pi) q[31];
U1q(0.806422718775965*pi,0.9709700003148543*pi) q[32];
U1q(0.664046219150519*pi,0.9769891316695496*pi) q[33];
U1q(0.39356167900373*pi,0.8768002255165603*pi) q[34];
U1q(0.223349517035646*pi,0.45633422699945037*pi) q[35];
U1q(0.734892678126096*pi,0.13916121795592007*pi) q[36];
U1q(0.631025993176563*pi,1.075755755537946*pi) q[37];
U1q(0.541881817846484*pi,1.2981358531631475*pi) q[38];
U1q(0.301663580510243*pi,1.7296121195247398*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[27],q[16];
RZZ(0.5*pi) q[17],q[31];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[37],q[23];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[26],q[33];
U1q(0.362079603303782*pi,0.7479890867071903*pi) q[0];
U1q(0.680586228005242*pi,0.5150967566271198*pi) q[1];
U1q(0.646431374092105*pi,0.9308594906421597*pi) q[2];
U1q(0.668830507422998*pi,1.8528170804462407*pi) q[3];
U1q(0.406894536083755*pi,1.7863209729107599*pi) q[4];
U1q(0.514825606454609*pi,0.41053309127690696*pi) q[5];
U1q(0.467228740931798*pi,0.6807230708675256*pi) q[6];
U1q(0.440608633887577*pi,1.1644863899671858*pi) q[7];
U1q(0.248358375007267*pi,1.2603336985917908*pi) q[8];
U1q(0.205651093113178*pi,0.6108686474723166*pi) q[9];
U1q(0.495285913776929*pi,1.5560316684251783*pi) q[10];
U1q(0.383787139274286*pi,1.780020806599608*pi) q[11];
U1q(0.374507195144595*pi,0.3389520724254398*pi) q[12];
U1q(0.260927257676028*pi,1.6247592263069999*pi) q[13];
U1q(0.209154663855561*pi,1.2377698079521195*pi) q[14];
U1q(0.495474714763919*pi,0.07554928681283002*pi) q[15];
U1q(0.928446368274012*pi,1.111046104831786*pi) q[16];
U1q(0.556931019395886*pi,0.6450024166165651*pi) q[17];
U1q(0.814467886816876*pi,0.8895989219227403*pi) q[18];
U1q(0.0879615115822774*pi,1.2345342558901393*pi) q[19];
U1q(0.665701371382651*pi,1.6687899316687096*pi) q[20];
U1q(0.368167419873029*pi,0.7631678151640209*pi) q[21];
U1q(0.639745127027704*pi,0.8450664649761324*pi) q[22];
U1q(0.143694156354879*pi,0.891536503707484*pi) q[23];
U1q(0.380699933479264*pi,1.0625559662865989*pi) q[24];
U1q(0.600401214959321*pi,1.3665140489279404*pi) q[25];
U1q(0.51139721019109*pi,1.00555921392845*pi) q[26];
U1q(0.876735193870709*pi,0.5030226983685226*pi) q[27];
U1q(0.400571905741898*pi,0.9130535148163688*pi) q[28];
U1q(0.149819170200139*pi,0.0786356819713454*pi) q[29];
U1q(0.293009833639677*pi,0.9269490873743633*pi) q[30];
U1q(0.781485863993352*pi,0.42368350356721995*pi) q[31];
U1q(0.39487227674725*pi,0.012661904882554254*pi) q[32];
U1q(0.63208859284391*pi,0.45404315348304003*pi) q[33];
U1q(0.383453964314692*pi,1.3118230995408293*pi) q[34];
U1q(0.49456689769152*pi,1.861920202240519*pi) q[35];
U1q(0.375695137904432*pi,1.0558239747649596*pi) q[36];
U1q(0.884877393407757*pi,1.95353793911557*pi) q[37];
U1q(0.160328279424238*pi,1.033137835314749*pi) q[38];
U1q(0.157256163743743*pi,0.81486041356544*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[39];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[27],q[9];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[22],q[14];
RZZ(0.5*pi) q[36],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[19],q[29];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[32],q[33];
U1q(0.457278911336713*pi,1.9659876150147007*pi) q[0];
U1q(0.468398474481235*pi,0.4098748187287704*pi) q[1];
U1q(0.513758488628027*pi,1.9683845112316298*pi) q[2];
U1q(0.56914955144862*pi,0.08545614626017084*pi) q[3];
U1q(0.450955621284546*pi,0.8876139597803405*pi) q[4];
U1q(0.734295197310809*pi,0.2622570200165768*pi) q[5];
U1q(0.375603376817536*pi,0.4029400177658857*pi) q[6];
U1q(0.573473242608398*pi,1.5916330840634068*pi) q[7];
U1q(0.694718342085186*pi,0.2222525911633113*pi) q[8];
U1q(0.517780841723721*pi,0.025963522229776714*pi) q[9];
U1q(0.516595084266629*pi,1.4571773882641477*pi) q[10];
U1q(0.263073701381412*pi,0.7903671662082781*pi) q[11];
U1q(0.637195037403892*pi,0.8183294387307098*pi) q[12];
U1q(0.737715287611789*pi,1.9404951637201204*pi) q[13];
U1q(0.270995186986608*pi,1.8957566063236513*pi) q[14];
U1q(0.659162292436731*pi,0.5173990729078497*pi) q[15];
U1q(0.731062475561095*pi,1.1841797368566969*pi) q[16];
U1q(0.454569693187285*pi,0.09660225361049513*pi) q[17];
U1q(0.747740500695876*pi,1.2948104243791505*pi) q[18];
U1q(0.341923518680806*pi,1.3672768503683006*pi) q[19];
U1q(0.28166765548654*pi,0.003507891805890395*pi) q[20];
U1q(0.255954518482923*pi,0.8987490634920992*pi) q[21];
U1q(0.226684003129837*pi,0.059193691618792954*pi) q[22];
U1q(0.275616557939363*pi,1.116576496113444*pi) q[23];
U1q(0.505393235807667*pi,1.2792877749625582*pi) q[24];
U1q(0.891223349518525*pi,1.18553878265352*pi) q[25];
U1q(0.564124917058875*pi,0.6745602527607799*pi) q[26];
U1q(0.564611020535365*pi,0.30447322973182267*pi) q[27];
U1q(0.13431521654463*pi,0.6760912680119482*pi) q[28];
U1q(0.189051571011845*pi,0.17288958761226603*pi) q[29];
U1q(0.788763149565696*pi,0.18407426975700503*pi) q[30];
U1q(0.827729392515999*pi,1.8064053183090198*pi) q[31];
U1q(0.679263933357092*pi,0.23938391964440342*pi) q[32];
U1q(0.520399818735171*pi,0.5640052376453095*pi) q[33];
U1q(0.24578955802676*pi,0.7523620471907009*pi) q[34];
U1q(0.380471319582301*pi,1.2239415514489487*pi) q[35];
U1q(0.849566618802612*pi,0.4554630944923703*pi) q[36];
U1q(0.3192966061064*pi,1.6002288104669198*pi) q[37];
U1q(0.421052007933974*pi,1.3377932781942192*pi) q[38];
U1q(0.672470003994351*pi,0.4357727744158897*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[5],q[33];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[22],q[15];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[37],q[31];
RZZ(0.5*pi) q[38],q[36];
U1q(0.736935671604859*pi,0.2445742012588994*pi) q[0];
U1q(0.448805215861104*pi,0.5786204281862997*pi) q[1];
U1q(0.655675737403859*pi,0.2703588899853493*pi) q[2];
U1q(0.158763279863817*pi,1.8173531215747705*pi) q[3];
U1q(0.769067696380533*pi,0.47311360784546963*pi) q[4];
U1q(0.117848071505199*pi,1.2668411869131777*pi) q[5];
U1q(0.653980365257799*pi,0.5639459963815945*pi) q[6];
U1q(0.484573598356026*pi,1.0373445391607063*pi) q[7];
U1q(0.616904189572703*pi,0.8918099217589628*pi) q[8];
U1q(0.705599998936139*pi,1.6582912139932269*pi) q[9];
U1q(0.08261521622582*pi,1.1588858029413789*pi) q[10];
U1q(0.614500923358267*pi,0.45826494617892877*pi) q[11];
U1q(0.357095100759975*pi,0.13681915456962912*pi) q[12];
U1q(0.609689042932139*pi,0.28636062320572986*pi) q[13];
U1q(0.483812042192268*pi,1.537194841704249*pi) q[14];
U1q(0.63468076824144*pi,0.9607728882636*pi) q[15];
U1q(0.637901354984673*pi,1.8631851185593167*pi) q[16];
U1q(0.701863744082855*pi,1.7824033709708953*pi) q[17];
U1q(0.923577552423501*pi,1.1896305788350006*pi) q[18];
U1q(0.673471591425318*pi,0.9059618719933997*pi) q[19];
U1q(0.628444019097291*pi,0.001382477358280454*pi) q[20];
U1q(0.799408541052751*pi,0.6702595975432999*pi) q[21];
U1q(0.554414826827312*pi,0.2957810183413532*pi) q[22];
U1q(0.928526907035416*pi,0.004043962454634453*pi) q[23];
U1q(0.298790679750529*pi,1.0687715835614586*pi) q[24];
U1q(0.583209916660863*pi,0.6448792424527507*pi) q[25];
U1q(0.466300573499177*pi,0.1469183668171805*pi) q[26];
U1q(0.667985472417678*pi,1.1763698301277241*pi) q[27];
U1q(0.524574792113285*pi,1.2999887907956484*pi) q[28];
U1q(0.485770459891391*pi,1.6521152568531061*pi) q[29];
U1q(0.698066121082949*pi,0.03190107484347493*pi) q[30];
U1q(0.396636154446682*pi,0.33254691951091964*pi) q[31];
U1q(0.336556624277941*pi,1.7089350261131635*pi) q[32];
U1q(0.175745717176717*pi,0.3371987108053993*pi) q[33];
U1q(0.325725547563627*pi,1.7806651781586993*pi) q[34];
U1q(0.494123920970934*pi,0.7026383410758488*pi) q[35];
U1q(0.517541144682613*pi,0.8792740843632796*pi) q[36];
U1q(0.503097080537065*pi,0.44794012364124036*pi) q[37];
U1q(0.24724231392672*pi,0.21739966785071907*pi) q[38];
U1q(0.691070571654772*pi,0.7995179085077702*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[12],q[38];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[29],q[33];
U1q(0.361407541862331*pi,1.035577713930799*pi) q[0];
U1q(0.22033653579558*pi,1.6203934738268995*pi) q[1];
U1q(0.599807763062939*pi,0.4637615583244994*pi) q[2];
U1q(0.541314779436264*pi,0.701430528100671*pi) q[3];
U1q(0.673870589653448*pi,0.8887609886849503*pi) q[4];
U1q(0.396976019312919*pi,0.2209253640286768*pi) q[5];
U1q(0.491221726620936*pi,1.2124064097040943*pi) q[6];
U1q(0.194192084126725*pi,0.9909580196826067*pi) q[7];
U1q(0.0964486186206051*pi,1.553669771893862*pi) q[8];
U1q(0.563290828161326*pi,0.46634575248862653*pi) q[9];
U1q(0.598492722203011*pi,0.09289778906937762*pi) q[10];
U1q(0.786969836788665*pi,0.39406683944852716*pi) q[11];
U1q(0.760552213601024*pi,0.5061740349438004*pi) q[12];
U1q(0.306609490935406*pi,0.7440698149596994*pi) q[13];
U1q(0.376505200220427*pi,1.5406855169993499*pi) q[14];
U1q(0.166099194012299*pi,0.028150894908000268*pi) q[15];
U1q(0.630685521969078*pi,1.731260178131917*pi) q[16];
U1q(0.271173168642729*pi,1.8530177435422956*pi) q[17];
U1q(0.225642099201894*pi,1.1366939596769008*pi) q[18];
U1q(0.666290260645561*pi,0.6853188752491999*pi) q[19];
U1q(0.257344207942076*pi,1.1500052626038997*pi) q[20];
U1q(0.323013935908627*pi,1.3035961708310992*pi) q[21];
U1q(0.866664043389316*pi,1.3566829685811026*pi) q[22];
U1q(0.582499568184439*pi,0.8216239857634342*pi) q[23];
U1q(0.468455634118271*pi,1.695862223487259*pi) q[24];
U1q(0.610664433670781*pi,0.7031681931740295*pi) q[25];
U1q(0.80303245026137*pi,1.0138909428504803*pi) q[26];
U1q(0.782577815178436*pi,0.20407848368372328*pi) q[27];
U1q(0.482483497047267*pi,0.2949140438633471*pi) q[28];
U1q(0.802770257472583*pi,0.055342054720107114*pi) q[29];
U1q(0.711720440083077*pi,1.1081959850053735*pi) q[30];
U1q(0.459361505046864*pi,1.6916330384585994*pi) q[31];
U1q(0.5128501363159*pi,0.5212781065551653*pi) q[32];
U1q(0.604369731603121*pi,1.5942848114365997*pi) q[33];
U1q(0.143148736907888*pi,1.6577030374368*pi) q[34];
U1q(0.754372976571858*pi,0.09827406232594882*pi) q[35];
U1q(0.345219690692482*pi,1.7660507342977994*pi) q[36];
U1q(0.407377829982561*pi,0.2848557924717898*pi) q[37];
U1q(0.428185059792112*pi,0.48227581732881575*pi) q[38];
U1q(0.80897174548088*pi,1.2153512789343992*pi) q[39];
rz(0.8388167495408005*pi) q[0];
rz(3.2749695216221006*pi) q[1];
rz(1.7796763410077094*pi) q[2];
rz(1.6867762693463284*pi) q[3];
rz(2.778062511937449*pi) q[4];
rz(2.151368808301921*pi) q[5];
rz(3.6812103109061063*pi) q[6];
rz(3.7918954766684934*pi) q[7];
rz(2.1873541065645377*pi) q[8];
rz(0.4452946139357721*pi) q[9];
rz(3.636595830025321*pi) q[10];
rz(0.7595119547457614*pi) q[11];
rz(0.6456597948370995*pi) q[12];
rz(2.4497160307479007*pi) q[13];
rz(3.9635567496296495*pi) q[14];
rz(3.6890561657224*pi) q[15];
rz(2.988388827222183*pi) q[16];
rz(2.398976396311305*pi) q[17];
rz(3.9516330824174997*pi) q[18];
rz(1.0336973826786995*pi) q[19];
rz(1.1732069589728003*pi) q[20];
rz(0.9654532840222991*pi) q[21];
rz(3.496698205926977*pi) q[22];
rz(1.1794688210815654*pi) q[23];
rz(3.080481590709441*pi) q[24];
rz(1.5215873587056894*pi) q[25];
rz(0.8766360059291198*pi) q[26];
rz(0.09576064294857645*pi) q[27];
rz(0.2135963752706509*pi) q[28];
rz(3.4142916072147926*pi) q[29];
rz(2.1917518603155273*pi) q[30];
rz(2.1522841559574992*pi) q[31];
rz(0.44129624989833616*pi) q[32];
rz(3.3181394585168*pi) q[33];
rz(1.0155093544513996*pi) q[34];
rz(2.1056681087274516*pi) q[35];
rz(3.1952794659208994*pi) q[36];
rz(2.3860526382808*pi) q[37];
rz(2.048855703745584*pi) q[38];
rz(2.0223399636801*pi) q[39];
U1q(1.36140754186233*pi,0.874394463471563*pi) q[0];
U1q(0.22033653579558*pi,1.895362995448945*pi) q[1];
U1q(0.599807763062939*pi,1.24343789933221*pi) q[2];
U1q(1.54131477943626*pi,1.388206797446982*pi) q[3];
U1q(0.673870589653448*pi,0.666823500622369*pi) q[4];
U1q(0.396976019312919*pi,1.372294172330625*pi) q[5];
U1q(0.491221726620936*pi,1.893616720610207*pi) q[6];
U1q(0.194192084126725*pi,1.782853496351126*pi) q[7];
U1q(3.096448618620605*pi,0.74102387845835*pi) q[8];
U1q(1.56329082816133*pi,1.9116403664243455*pi) q[9];
U1q(0.598492722203011*pi,0.729493619094705*pi) q[10];
U1q(0.786969836788665*pi,0.153578794194276*pi) q[11];
U1q(0.760552213601024*pi,0.151833829780862*pi) q[12];
U1q(1.30660949093541*pi,0.193785845707575*pi) q[13];
U1q(0.376505200220427*pi,0.504242266628982*pi) q[14];
U1q(0.166099194012299*pi,0.71720706063041*pi) q[15];
U1q(3.630685521969078*pi,1.719649005354023*pi) q[16];
U1q(1.27117316864273*pi,1.251994139853654*pi) q[17];
U1q(0.225642099201894*pi,0.0883270420943967*pi) q[18];
U1q(0.666290260645561*pi,0.719016257927918*pi) q[19];
U1q(1.25734420794208*pi,1.323212221576639*pi) q[20];
U1q(1.32301393590863*pi,1.269049454853387*pi) q[21];
U1q(1.86666404338932*pi,1.853381174508058*pi) q[22];
U1q(0.582499568184439*pi,1.00109280684496*pi) q[23];
U1q(0.468455634118271*pi,1.776343814196689*pi) q[24];
U1q(1.61066443367078*pi,1.22475555187972*pi) q[25];
U1q(3.803032450261371*pi,0.890526948779549*pi) q[26];
U1q(1.78257781517844*pi,1.29983912663232*pi) q[27];
U1q(0.482483497047267*pi,1.508510419134*pi) q[28];
U1q(3.802770257472583*pi,0.469633661934868*pi) q[29];
U1q(0.711720440083077*pi,0.299947845320996*pi) q[30];
U1q(1.45936150504686*pi,0.843917194416034*pi) q[31];
U1q(1.5128501363159*pi,1.9625743564535012*pi) q[32];
U1q(1.60436973160312*pi,1.9124242699533833*pi) q[33];
U1q(0.143148736907888*pi,1.673212391888267*pi) q[34];
U1q(1.75437297657186*pi,1.20394217105336*pi) q[35];
U1q(1.34521969069248*pi,1.9613302002187545*pi) q[36];
U1q(1.40737782998256*pi,1.6709084307526019*pi) q[37];
U1q(0.428185059792112*pi,1.531131521074476*pi) q[38];
U1q(0.80897174548088*pi,0.237691242614454*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[5],q[2];
RZZ(0.5*pi) q[6],q[3];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[35],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[10],q[27];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[12],q[38];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[22],q[26];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[29],q[33];
U1q(3.2630643283951413*pi,1.6653979761434377*pi) q[0];
U1q(0.448805215861104*pi,1.85358994980836*pi) q[1];
U1q(0.655675737403859*pi,0.0500352309930677*pi) q[2];
U1q(3.158763279863817*pi,0.2722842039728761*pi) q[3];
U1q(0.769067696380533*pi,0.2511761197829099*pi) q[4];
U1q(3.117848071505199*pi,1.418209995215092*pi) q[5];
U1q(1.6539803652578*pi,1.245156307287698*pi) q[6];
U1q(3.484573598356026*pi,1.82924001582921*pi) q[7];
U1q(3.383095810427297*pi,1.402883728593184*pi) q[8];
U1q(3.294400001063861*pi,1.7196949049197274*pi) q[9];
U1q(1.08261521622582*pi,1.795481632966654*pi) q[10];
U1q(0.614500923358267*pi,1.217776900924689*pi) q[11];
U1q(1.35709510075997*pi,0.7824789494067299*pi) q[12];
U1q(3.390310957067861*pi,1.6514950374615465*pi) q[13];
U1q(1.48381204219227*pi,0.50075159133387*pi) q[14];
U1q(1.63468076824144*pi,0.6498290539860101*pi) q[15];
U1q(3.362098645015327*pi,0.5877240649266335*pi) q[16];
U1q(3.2981362559171448*pi,0.3226085124250426*pi) q[17];
U1q(0.923577552423501*pi,0.14126366125249*pi) q[18];
U1q(1.67347159142532*pi,0.9396592546721201*pi) q[19];
U1q(3.3715559809027082*pi,1.471835006822234*pi) q[20];
U1q(3.20059145894725*pi,1.902386028141176*pi) q[21];
U1q(1.55441482682731*pi,1.9142831247478123*pi) q[22];
U1q(1.92852690703542*pi,0.183512783536226*pi) q[23];
U1q(1.29879067975053*pi,0.1492531742709402*pi) q[24];
U1q(1.58320991666086*pi,1.283044502601*pi) q[25];
U1q(3.533699426500823*pi,1.7574995248128027*pi) q[26];
U1q(3.332014527582321*pi,1.3275477801882662*pi) q[27];
U1q(0.524574792113285*pi,0.513585166066246*pi) q[28];
U1q(3.5142295401086088*pi,0.8728604598018297*pi) q[29];
U1q(0.698066121082949*pi,1.2236529351590621*pi) q[30];
U1q(3.396636154446682*pi,0.20300331336369193*pi) q[31];
U1q(3.336556624277941*pi,1.77491743689543*pi) q[32];
U1q(1.17574571717672*pi,0.1695103705845904*pi) q[33];
U1q(0.325725547563627*pi,0.7961745326101299*pi) q[34];
U1q(3.505876079029066*pi,0.5995778923034414*pi) q[35];
U1q(1.51754114468261*pi,0.8481068501532809*pi) q[36];
U1q(3.4969029194629337*pi,1.5078240995831482*pi) q[37];
U1q(1.24724231392672*pi,0.2662553715963101*pi) q[38];
U1q(0.691070571654772*pi,0.82185787218785*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[5],q[33];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[11],q[16];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[22],q[15];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[35],q[28];
RZZ(0.5*pi) q[37],q[31];
RZZ(0.5*pi) q[38],q[36];
U1q(3.457278911336713*pi,0.9439845623876291*pi) q[0];
U1q(1.46839847448123*pi,1.6848443403508204*pi) q[1];
U1q(1.51375848862803*pi,1.748060852239342*pi) q[2];
U1q(1.56914955144862*pi,0.5403872286582843*pi) q[3];
U1q(1.45095562128455*pi,0.6656764717177701*pi) q[4];
U1q(1.73429519731081*pi,0.4227941621116331*pi) q[5];
U1q(1.37560337681754*pi,0.40616228590340686*pi) q[6];
U1q(1.5734732426084*pi,0.2749514709265317*pi) q[7];
U1q(3.305281657914814*pi,0.07244105918886812*pi) q[8];
U1q(3.482219158276279*pi,0.3520225966831585*pi) q[9];
U1q(3.516595084266629*pi,0.4971900476438391*pi) q[10];
U1q(0.263073701381412*pi,1.54987912095404*pi) q[11];
U1q(1.63719503740389*pi,0.10096866524564385*pi) q[12];
U1q(1.73771528761179*pi,0.9973604969471516*pi) q[13];
U1q(3.729004813013392*pi,1.142189826714462*pi) q[14];
U1q(1.65916229243673*pi,1.0932028693417628*pi) q[15];
U1q(1.7310624755611*pi,0.2667294466292063*pi) q[16];
U1q(3.545430306812715*pi,1.008409629785493*pi) q[17];
U1q(1.74774050069588*pi,1.2464435067966102*pi) q[18];
U1q(3.658076481319193*pi,1.4783442762972734*pi) q[19];
U1q(3.718332344513459*pi,0.46970959237462484*pi) q[20];
U1q(3.744045481517077*pi,0.6738965621923967*pi) q[21];
U1q(0.226684003129837*pi,1.6776957980252523*pi) q[22];
U1q(3.724383442060637*pi,1.0709802498774526*pi) q[23];
U1q(1.50539323580767*pi,0.9387369828698571*pi) q[24];
U1q(0.891223349518525*pi,1.8237040428017701*pi) q[25];
U1q(3.435875082941125*pi,1.2298576388692224*pi) q[26];
U1q(3.435388979464635*pi,0.19944438058421238*pi) q[27];
U1q(1.13431521654463*pi,1.889687643282599*pi) q[28];
U1q(1.18905157101185*pi,0.35208612904267866*pi) q[29];
U1q(1.7887631495657*pi,1.3758261300725771*pi) q[30];
U1q(1.827729392516*pi,1.6768617121618*pi) q[31];
U1q(3.679263933357093*pi,1.305366330426635*pi) q[32];
U1q(1.52039981873517*pi,0.3963168974244906*pi) q[33];
U1q(0.24578955802676*pi,1.7678714016421302*pi) q[34];
U1q(3.619528680417699*pi,0.0782746819303094*pi) q[35];
U1q(0.849566618802612*pi,0.42429586028236344*pi) q[36];
U1q(3.6807033938936*pi,0.35553541275746836*pi) q[37];
U1q(3.421052007933974*pi,1.1458617612527986*pi) q[38];
U1q(1.67247000399435*pi,0.45811273809597974*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[1],q[13];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[39];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[27],q[9];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[22],q[14];
RZZ(0.5*pi) q[36],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[19],q[29];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[32],q[33];
U1q(0.362079603303782*pi,1.7259860340800728*pi) q[0];
U1q(1.68058622800524*pi,1.579622402452471*pi) q[1];
U1q(3.353568625907895*pi,1.7855858728288145*pi) q[2];
U1q(1.668830507423*pi,1.773026294472209*pi) q[3];
U1q(3.593105463916245*pi,0.7669694585873517*pi) q[4];
U1q(3.514825606454609*pi,0.5710702333719411*pi) q[5];
U1q(0.467228740931798*pi,0.6839453390050472*pi) q[6];
U1q(0.440608633887577*pi,1.8478047768303218*pi) q[7];
U1q(3.751641624992733*pi,0.03435995176038409*pi) q[8];
U1q(1.20565109311318*pi,1.7671174714406104*pi) q[9];
U1q(1.49528591377693*pi,0.596044327804869*pi) q[10];
U1q(1.38378713927429*pi,1.5395327613453702*pi) q[11];
U1q(1.37450719514459*pi,0.621591298940364*pi) q[12];
U1q(0.260927257676028*pi,0.6816245595340302*pi) q[13];
U1q(3.790845336144438*pi,1.8001766250859617*pi) q[14];
U1q(0.495474714763919*pi,1.6513530832467431*pi) q[15];
U1q(1.92844636827401*pi,0.1935958146043002*pi) q[16];
U1q(3.443068980604114*pi,0.4600094667794128*pi) q[17];
U1q(1.81446788681688*pi,0.6516550092530231*pi) q[18];
U1q(3.087961511582277*pi,0.6110868707754236*pi) q[19];
U1q(3.334298628617349*pi,1.804427552511795*pi) q[20];
U1q(3.63183258012697*pi,0.8094778105204563*pi) q[21];
U1q(0.639745127027704*pi,0.4635685713825921*pi) q[22];
U1q(3.856305843645121*pi,1.2960202422834024*pi) q[23];
U1q(1.38069993347926*pi,0.7220051741938773*pi) q[24];
U1q(0.600401214959321*pi,1.004679309076188*pi) q[25];
U1q(3.51139721019109*pi,0.8988586777015686*pi) q[26];
U1q(1.87673519387071*pi,0.0008949119475222256*pi) q[27];
U1q(3.400571905741899*pi,0.6527253964782029*pi) q[28];
U1q(1.14981917020014*pi,0.25783222340175893*pi) q[29];
U1q(1.29300983363968*pi,0.6329513124552201*pi) q[30];
U1q(3.218514136006648*pi,0.05958352690359954*pi) q[31];
U1q(3.6051277232527488*pi,1.532088345188484*pi) q[32];
U1q(1.63208859284391*pi,1.5062789815867577*pi) q[33];
U1q(1.38345396431469*pi,1.3273324539922404*pi) q[34];
U1q(1.49456689769152*pi,0.44029603113876764*pi) q[35];
U1q(0.375695137904432*pi,0.024656740554956258*pi) q[36];
U1q(3.115122606592244*pi,1.0022262841088185*pi) q[37];
U1q(1.16032827942424*pi,0.8412063183733487*pi) q[38];
U1q(3.842743836256256*pi,0.07902509894644183*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[7],q[2];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[28],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[10],q[30];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[27],q[16];
RZZ(0.5*pi) q[17],q[31];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[37],q[23];
RZZ(0.5*pi) q[24],q[39];
RZZ(0.5*pi) q[26],q[33];
U1q(0.188084487236584*pi,1.153024767434375*pi) q[0];
U1q(3.124871135822001*pi,0.8484787473422202*pi) q[1];
U1q(3.369264160353686*pi,0.8482022838186367*pi) q[2];
U1q(1.62357351057354*pi,0.005505542073039216*pi) q[3];
U1q(3.047363318172348*pi,1.5105231266278356*pi) q[4];
U1q(1.56259082951608*pi,0.1363603427078548*pi) q[5];
U1q(1.55928500709476*pi,0.718600025840717*pi) q[6];
U1q(1.51317547721543*pi,0.9168699521878416*pi) q[7];
U1q(3.751401296807305*pi,1.749835565121824*pi) q[8];
U1q(0.451425641012476*pi,0.7395895652586297*pi) q[9];
U1q(3.70613479625443*pi,1.8703165214933595*pi) q[10];
U1q(1.73573253932976*pi,0.4883489335358684*pi) q[11];
U1q(1.53700775741441*pi,0.9636653243881952*pi) q[12];
U1q(1.81684592291077*pi,0.9551891928032985*pi) q[13];
U1q(1.5045367699798*pi,1.7543037795265919*pi) q[14];
U1q(1.76561110463457*pi,1.3856847903377423*pi) q[15];
U1q(1.4701829402508*pi,0.610528328640962*pi) q[16];
U1q(1.57555303980954*pi,1.1589013646757804*pi) q[17];
U1q(0.341904937113912*pi,0.3542269576681827*pi) q[18];
U1q(1.43404656825589*pi,1.6577450384019041*pi) q[19];
U1q(3.493044767652049*pi,0.7812086595680849*pi) q[20];
U1q(3.099337914460862*pi,0.6445767833784366*pi) q[21];
U1q(0.702379324046984*pi,0.7260015706269325*pi) q[22];
U1q(3.199554818866256*pi,0.6987042513673425*pi) q[23];
U1q(1.53872079319435*pi,0.8658817608041653*pi) q[24];
U1q(1.48030567498428*pi,1.8741062108882611*pi) q[25];
U1q(1.7518283945819*pi,1.7811469935640387*pi) q[26];
U1q(1.34321999898934*pi,1.0829184544871473*pi) q[27];
U1q(0.770671362860269*pi,0.7929487361283529*pi) q[28];
U1q(1.21399721722491*pi,1.351573750921374*pi) q[29];
U1q(1.29491868878914*pi,0.757098437841617*pi) q[30];
U1q(1.90801362936004*pi,0.5341642426972255*pi) q[31];
U1q(1.80642271877597*pi,1.5737802497561848*pi) q[32];
U1q(0.664046219150519*pi,1.0292249597732672*pi) q[33];
U1q(3.3935616790037297*pi,1.762355328016504*pi) q[34];
U1q(0.223349517035646*pi,0.03471005589769871*pi) q[35];
U1q(1.7348926781261*pi,1.1079939837459163*pi) q[36];
U1q(3.368974006823437*pi,0.8800084676864488*pi) q[37];
U1q(1.54188181784648*pi,1.5762083005249554*pi) q[38];
U1q(3.698336419489756*pi,0.16427339298714116*pi) q[39];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[1],q[4];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[38],q[10];
RZZ(0.5*pi) q[14],q[11];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[27],q[19];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[35],q[39];
U1q(1.8596202836695*pi,0.4342736234416291*pi) q[0];
U1q(3.334068042967539*pi,0.4096826030678269*pi) q[1];
U1q(3.679318247831059*pi,1.5671203722547262*pi) q[2];
U1q(3.528692190816284*pi,1.4795483720979714*pi) q[3];
U1q(3.652477603578029*pi,0.8068934015485447*pi) q[4];
U1q(0.593483072391106*pi,1.5638454930825851*pi) q[5];
U1q(3.216224354134524*pi,0.7828223501475549*pi) q[6];
U1q(3.332065162672106*pi,0.8375098748900021*pi) q[7];
U1q(3.7872375006405212*pi,1.0642570449111943*pi) q[8];
U1q(1.36994144485094*pi,1.8535615244749497*pi) q[9];
U1q(1.48290321725938*pi,0.131436253971688*pi) q[10];
U1q(0.705335715546129*pi,1.1835289559138777*pi) q[11];
U1q(1.64678722430446*pi,0.8012587653473746*pi) q[12];
U1q(1.45943305867663*pi,1.9860105746717132*pi) q[13];
U1q(1.87282989923521*pi,0.2725008561803821*pi) q[14];
U1q(1.69771233511049*pi,1.8182967702083825*pi) q[15];
U1q(1.92583523419417*pi,0.569821646557612*pi) q[16];
U1q(1.50279296466406*pi,0.27385736919045023*pi) q[17];
U1q(1.90214082605*pi,0.2798568035470934*pi) q[18];
U1q(3.545130020791594*pi,0.03983603901123356*pi) q[19];
U1q(3.312580474357863*pi,0.9087960091505358*pi) q[20];
U1q(3.760773980407991*pi,0.9828487452308767*pi) q[21];
U1q(0.798202299138304*pi,1.9001725465792223*pi) q[22];
U1q(3.761622862062429*pi,1.8365248347064393*pi) q[23];
U1q(0.724414118748761*pi,0.31609666359721533*pi) q[24];
U1q(1.47182902841111*pi,0.20549052262501913*pi) q[25];
U1q(3.523409865580471*pi,1.4297972097380214*pi) q[26];
U1q(3.661369696164319*pi,1.5364338652035388*pi) q[27];
U1q(1.23712393186368*pi,0.8579396005173336*pi) q[28];
U1q(1.19286725685178*pi,1.841259713658884*pi) q[29];
U1q(1.75245239055162*pi,1.4793527665520168*pi) q[30];
U1q(0.374876616676999*pi,1.079825910596071*pi) q[31];
U1q(1.10045179499242*pi,0.08473734869972382*pi) q[32];
U1q(1.7826393018133*pi,1.194812412065117*pi) q[33];
U1q(0.180252335136177*pi,0.7490960115778638*pi) q[34];
U1q(1.34002487534904*pi,0.7372250335148989*pi) q[35];
U1q(3.3462877872449592*pi,1.3174405477710787*pi) q[36];
U1q(1.55283126564084*pi,1.8007369275067475*pi) q[37];
U1q(0.272568555917096*pi,1.113153789489095*pi) q[38];
U1q(3.224827533431626*pi,0.8393587471028567*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[8],q[15];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[17],q[26];
RZZ(0.5*pi) q[28],q[19];
RZZ(0.5*pi) q[20],q[34];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[31],q[39];
RZZ(0.5*pi) q[35],q[38];
U1q(1.13287871109974*pi,1.8560358374456474*pi) q[0];
U1q(1.35921237945345*pi,0.2859648949433451*pi) q[1];
U1q(1.33031405252093*pi,0.4104849267031283*pi) q[2];
U1q(1.36161423976044*pi,0.536453855512832*pi) q[3];
U1q(1.51004624585604*pi,1.3529004034623588*pi) q[4];
U1q(0.441437490495472*pi,0.8259503189879354*pi) q[5];
U1q(1.84819830605081*pi,1.1990526714871135*pi) q[6];
U1q(0.579899404375027*pi,0.8621597057531618*pi) q[7];
U1q(3.426291088722548*pi,0.5338959856395142*pi) q[8];
U1q(1.31104486165108*pi,1.9828418016228815*pi) q[9];
U1q(0.724497639462827*pi,0.45665176973410837*pi) q[10];
U1q(0.490085517157544*pi,1.4579675966839876*pi) q[11];
U1q(1.75637168259918*pi,1.0396731163911301*pi) q[12];
U1q(0.382880021667634*pi,0.10706420096691316*pi) q[13];
U1q(1.49782356286289*pi,1.1785393011764231*pi) q[14];
U1q(0.455119909929831*pi,0.03747230743075214*pi) q[15];
U1q(1.29275156863519*pi,0.31971794873127646*pi) q[16];
U1q(1.46814794233177*pi,0.605727438305605*pi) q[17];
U1q(3.860992421227862*pi,0.4365939629597291*pi) q[18];
U1q(1.72789206891479*pi,1.9288838133811987*pi) q[19];
U1q(0.900299959064042*pi,1.2367478643739558*pi) q[20];
U1q(1.40661697074917*pi,0.4294592110206663*pi) q[21];
U1q(0.678241906937981*pi,1.7081449262119817*pi) q[22];
U1q(1.41566698722016*pi,0.9896710189693612*pi) q[23];
U1q(0.188866177774875*pi,1.7737427108436155*pi) q[24];
U1q(0.594630062875135*pi,0.9864733355688071*pi) q[25];
U1q(1.5065603547358*pi,1.1790027754539776*pi) q[26];
U1q(1.50132998464332*pi,1.039080033955166*pi) q[27];
U1q(1.35667461925444*pi,1.605441627183585*pi) q[28];
U1q(1.69194227477596*pi,0.3459044725116067*pi) q[29];
U1q(0.33653438622163*pi,1.1956434682474328*pi) q[30];
U1q(0.430040705114152*pi,0.4392105792437313*pi) q[31];
U1q(1.42946966305784*pi,1.1589843769783565*pi) q[32];
U1q(1.36294639056355*pi,0.5875780964490431*pi) q[33];
U1q(0.63610615043126*pi,0.4044274393010454*pi) q[34];
U1q(1.93720304199315*pi,1.3516161498071955*pi) q[35];
U1q(3.42953286292293*pi,1.780720069866602*pi) q[36];
U1q(0.73217305303267*pi,1.2839466397876276*pi) q[37];
U1q(0.255356370400301*pi,1.7099947169622265*pi) q[38];
U1q(0.460754951705264*pi,1.818733896461648*pi) q[39];
rz(0.14396416255435263*pi) q[0];
rz(3.714035105056655*pi) q[1];
rz(1.5895150732968717*pi) q[2];
rz(1.463546144487168*pi) q[3];
rz(0.6470995965376412*pi) q[4];
rz(3.1740496810120646*pi) q[5];
rz(2.8009473285128865*pi) q[6];
rz(3.137840294246838*pi) q[7];
rz(3.4661040143604858*pi) q[8];
rz(2.0171581983771185*pi) q[9];
rz(1.5433482302658916*pi) q[10];
rz(0.5420324033160124*pi) q[11];
rz(0.9603268836088699*pi) q[12];
rz(3.892935799033087*pi) q[13];
rz(0.8214606988235769*pi) q[14];
rz(3.962527692569248*pi) q[15];
rz(3.6802820512687235*pi) q[16];
rz(3.394272561694395*pi) q[17];
rz(3.563406037040271*pi) q[18];
rz(0.07111618661880126*pi) q[19];
rz(2.763252135626044*pi) q[20];
rz(3.5705407889793337*pi) q[21];
rz(0.29185507378801834*pi) q[22];
rz(1.0103289810306388*pi) q[23];
rz(2.2262572891563845*pi) q[24];
rz(3.013526664431193*pi) q[25];
rz(2.8209972245460224*pi) q[26];
rz(0.960919966044834*pi) q[27];
rz(2.394558372816415*pi) q[28];
rz(3.6540955274883933*pi) q[29];
rz(0.8043565317525672*pi) q[30];
rz(3.5607894207562687*pi) q[31];
rz(0.8410156230216435*pi) q[32];
rz(1.4124219035509569*pi) q[33];
rz(1.5955725606989546*pi) q[34];
rz(2.6483838501928045*pi) q[35];
rz(2.219279930133398*pi) q[36];
rz(2.7160533602123724*pi) q[37];
rz(2.2900052830377735*pi) q[38];
rz(2.181266103538352*pi) q[39];
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
