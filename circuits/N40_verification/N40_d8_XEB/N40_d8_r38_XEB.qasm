OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.31229963773667*pi,0.551811279564922*pi) q[0];
U1q(0.381414196916836*pi,1.220263771877093*pi) q[1];
U1q(0.710293625347567*pi,0.139709863957204*pi) q[2];
U1q(0.90848925257548*pi,0.551040274039049*pi) q[3];
U1q(0.414905459295303*pi,0.342947704110199*pi) q[4];
U1q(0.21604798369471*pi,1.147265469507937*pi) q[5];
U1q(0.208783820822877*pi,0.7446401429166201*pi) q[6];
U1q(0.600721535272031*pi,1.658782052396434*pi) q[7];
U1q(0.713451066340055*pi,0.869419993810602*pi) q[8];
U1q(0.536460337105652*pi,1.839658380795983*pi) q[9];
U1q(0.913181469356353*pi,0.799099889741057*pi) q[10];
U1q(0.605824457295893*pi,0.29707445846147*pi) q[11];
U1q(0.596617373372141*pi,1.35810216365999*pi) q[12];
U1q(0.204752309847391*pi,1.895395532249792*pi) q[13];
U1q(0.59639226366404*pi,1.70761949312399*pi) q[14];
U1q(0.668274199912348*pi,1.42421155795641*pi) q[15];
U1q(0.312267263117166*pi,0.435374002678866*pi) q[16];
U1q(0.258701840250185*pi,0.0132852542740087*pi) q[17];
U1q(0.566042118057774*pi,0.112905534315515*pi) q[18];
U1q(0.444830197557442*pi,1.643472365724023*pi) q[19];
U1q(0.801794808544372*pi,1.9364724352676987*pi) q[20];
U1q(0.497818331109057*pi,0.442872020489385*pi) q[21];
U1q(0.6156048332197*pi,0.432440263323025*pi) q[22];
U1q(0.469752618102422*pi,0.158170688069793*pi) q[23];
U1q(0.616101790852775*pi,0.383535305765628*pi) q[24];
U1q(0.482124727343094*pi,1.6295287916342849*pi) q[25];
U1q(0.699989258685061*pi,0.279889244002472*pi) q[26];
U1q(0.144037017532434*pi,1.744178766528802*pi) q[27];
U1q(0.578497609318864*pi,1.6062152528426341*pi) q[28];
U1q(0.092728909829451*pi,1.827631162184435*pi) q[29];
U1q(0.847798571181012*pi,0.969621069400127*pi) q[30];
U1q(0.830326107665229*pi,1.1553992117491*pi) q[31];
U1q(0.342904616166809*pi,1.128577683599039*pi) q[32];
U1q(0.517303069627405*pi,1.8108624254409569*pi) q[33];
U1q(0.486573700733795*pi,0.433190499927211*pi) q[34];
U1q(0.33126218418634*pi,1.00945828393731*pi) q[35];
U1q(0.829090421306399*pi,1.9213960041640978*pi) q[36];
U1q(0.473531544079661*pi,1.274831626395457*pi) q[37];
U1q(0.259215984872314*pi,1.458252422236378*pi) q[38];
U1q(0.371402837592488*pi,0.367067097384413*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[3],q[30];
RZZ(0.5*pi) q[6],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[16],q[12];
RZZ(0.5*pi) q[14],q[26];
RZZ(0.5*pi) q[15],q[22];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[32],q[37];
RZZ(0.5*pi) q[33],q[36];
U1q(0.491023832196781*pi,0.85505880923437*pi) q[0];
U1q(0.675489311468797*pi,1.08651550120869*pi) q[1];
U1q(0.343338860976936*pi,1.4253186306584902*pi) q[2];
U1q(0.726143140530858*pi,1.4560167259458*pi) q[3];
U1q(0.698883137105947*pi,1.15408540790617*pi) q[4];
U1q(0.796357631444934*pi,0.8078242131288*pi) q[5];
U1q(0.358406503551781*pi,0.2764621403057501*pi) q[6];
U1q(0.376879160700361*pi,0.4395918437808599*pi) q[7];
U1q(0.596509500768079*pi,0.220067071874858*pi) q[8];
U1q(0.592853482904122*pi,1.7978857992746402*pi) q[9];
U1q(0.326387716574645*pi,1.95148207316964*pi) q[10];
U1q(0.643347315897806*pi,0.4774699330813099*pi) q[11];
U1q(0.753821145653017*pi,0.709203268897923*pi) q[12];
U1q(0.494562580178645*pi,1.9781374440278001*pi) q[13];
U1q(0.293303273391267*pi,0.137765752213174*pi) q[14];
U1q(0.331083837537727*pi,1.8887753588928096*pi) q[15];
U1q(0.20284169095828*pi,1.6265164369871101*pi) q[16];
U1q(0.42692492548534*pi,1.7937808645279398*pi) q[17];
U1q(0.20115716249478*pi,1.6168326748855*pi) q[18];
U1q(0.619673539842156*pi,0.4351018812779399*pi) q[19];
U1q(0.450269968025979*pi,0.5906616538235299*pi) q[20];
U1q(0.264602277131943*pi,1.93307507043101*pi) q[21];
U1q(0.300878335101893*pi,0.9203419970586*pi) q[22];
U1q(0.730870521716293*pi,1.78804348287838*pi) q[23];
U1q(0.564472172214138*pi,0.018028229170850008*pi) q[24];
U1q(0.516618979057923*pi,0.14854432098582993*pi) q[25];
U1q(0.378260655774614*pi,1.3435148744123602*pi) q[26];
U1q(0.898904431730394*pi,1.7686915367034701*pi) q[27];
U1q(0.530252032154004*pi,0.5998099416248102*pi) q[28];
U1q(0.40639144276699*pi,1.40286736611362*pi) q[29];
U1q(0.17740844959523*pi,0.4769543003055201*pi) q[30];
U1q(0.268272716540808*pi,0.2641307064584699*pi) q[31];
U1q(0.253508579708732*pi,1.75920610054244*pi) q[32];
U1q(0.821667837336177*pi,1.7923290039561102*pi) q[33];
U1q(0.700214406342756*pi,0.8198700559015399*pi) q[34];
U1q(0.233971550848753*pi,1.0364132014885303*pi) q[35];
U1q(0.554943714951313*pi,1.43086554213378*pi) q[36];
U1q(0.377459189113822*pi,0.7597233844100999*pi) q[37];
U1q(0.351323740388873*pi,1.7435822751999401*pi) q[38];
U1q(0.471407755236009*pi,1.4862423546433101*pi) q[39];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[16],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[37],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[17],q[9];
RZZ(0.5*pi) q[11],q[18];
RZZ(0.5*pi) q[24],q[12];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[36],q[15];
RZZ(0.5*pi) q[19],q[30];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[21],q[28];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[25],q[26];
RZZ(0.5*pi) q[35],q[31];
RZZ(0.5*pi) q[34],q[38];
U1q(0.212476998117787*pi,1.12037616195994*pi) q[0];
U1q(0.200538787093961*pi,0.69847904774028*pi) q[1];
U1q(0.509609066759494*pi,0.5095671812585998*pi) q[2];
U1q(0.417436056127029*pi,1.8976277478639103*pi) q[3];
U1q(0.44700275739795*pi,0.5673323837537501*pi) q[4];
U1q(0.238834367624638*pi,1.6018048198916999*pi) q[5];
U1q(0.303492812928038*pi,0.7428234836907901*pi) q[6];
U1q(0.332925620180565*pi,1.87234780937355*pi) q[7];
U1q(0.235801445615176*pi,1.9391107511254302*pi) q[8];
U1q(0.771752531968379*pi,1.8701133201685103*pi) q[9];
U1q(0.827430650721494*pi,0.38761834455488*pi) q[10];
U1q(0.187374136666678*pi,0.0028826118995599614*pi) q[11];
U1q(0.142465397248947*pi,1.236779951345244*pi) q[12];
U1q(0.367904527774299*pi,0.15749866218271968*pi) q[13];
U1q(0.563235633229589*pi,0.8406436709117999*pi) q[14];
U1q(0.657603032337479*pi,1.08216806539658*pi) q[15];
U1q(0.308862228683244*pi,0.2848446041036796*pi) q[16];
U1q(0.269088817563756*pi,1.8350104916861198*pi) q[17];
U1q(0.331369094604112*pi,0.5041400209251696*pi) q[18];
U1q(0.197396123290845*pi,1.6097244464126303*pi) q[19];
U1q(0.368709314402627*pi,0.3853915977131903*pi) q[20];
U1q(0.468234374870159*pi,0.3005562981788401*pi) q[21];
U1q(0.589451903577411*pi,0.7376746896281201*pi) q[22];
U1q(0.746236342352675*pi,1.43216744776363*pi) q[23];
U1q(0.173978670695127*pi,0.34001942662757*pi) q[24];
U1q(0.615222117236208*pi,1.3703040708653198*pi) q[25];
U1q(0.353457021100441*pi,0.9455080715195603*pi) q[26];
U1q(0.449389484923439*pi,1.4572559252937198*pi) q[27];
U1q(0.826170374303935*pi,0.9992095920100104*pi) q[28];
U1q(0.293228012210282*pi,0.16590459292152016*pi) q[29];
U1q(0.282022780594429*pi,1.9386630017129303*pi) q[30];
U1q(0.562192214812832*pi,1.9177150132196203*pi) q[31];
U1q(0.774288005289679*pi,0.11296630646202033*pi) q[32];
U1q(0.430873289723375*pi,1.69875814168531*pi) q[33];
U1q(0.0840674910232276*pi,1.0592431775683901*pi) q[34];
U1q(0.210324976786816*pi,1.96908976014869*pi) q[35];
U1q(0.748299321852672*pi,1.5242881552884597*pi) q[36];
U1q(0.447705605810701*pi,0.2626871148802099*pi) q[37];
U1q(0.653834851553764*pi,1.9388363731684697*pi) q[38];
U1q(0.329474835260938*pi,1.6004313988291399*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[24],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[4],q[23];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[25],q[7];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[36],q[12];
RZZ(0.5*pi) q[13],q[35];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[19],q[26];
RZZ(0.5*pi) q[37],q[21];
RZZ(0.5*pi) q[38],q[29];
U1q(0.387966254639142*pi,0.42642698292734993*pi) q[0];
U1q(0.670330401769195*pi,0.9914429563486706*pi) q[1];
U1q(0.493763955742203*pi,1.4425532157554501*pi) q[2];
U1q(0.031052821205373*pi,1.3703711309709377e-05*pi) q[3];
U1q(0.407450549324203*pi,0.9615000790192001*pi) q[4];
U1q(0.907084539295692*pi,0.13958847647263006*pi) q[5];
U1q(0.621083054875151*pi,1.9525546518199297*pi) q[6];
U1q(0.630620968435692*pi,0.37494965197582975*pi) q[7];
U1q(0.488283640350923*pi,1.74172177729998*pi) q[8];
U1q(0.440097916545129*pi,1.58576788595526*pi) q[9];
U1q(0.688884994642753*pi,0.2386743297851197*pi) q[10];
U1q(0.717635408788646*pi,0.0027326365296200805*pi) q[11];
U1q(0.281897162148384*pi,0.33051827491471997*pi) q[12];
U1q(0.596366498883061*pi,0.2138000227896999*pi) q[13];
U1q(0.869516658802477*pi,0.3768955038071997*pi) q[14];
U1q(0.207128212025903*pi,1.1067359210568197*pi) q[15];
U1q(0.5365431596982*pi,1.0547422954987704*pi) q[16];
U1q(0.648315659198168*pi,0.28212949969307033*pi) q[17];
U1q(0.837782486951361*pi,1.6509978895310606*pi) q[18];
U1q(0.63923100269769*pi,1.9035031754580398*pi) q[19];
U1q(0.643460236414826*pi,0.31632405968427957*pi) q[20];
U1q(0.499069996417119*pi,1.3869112108748105*pi) q[21];
U1q(0.622166833876429*pi,0.30904614536484987*pi) q[22];
U1q(0.454171070459383*pi,0.7460824979936804*pi) q[23];
U1q(0.737062244791592*pi,1.25615635921422*pi) q[24];
U1q(0.42320971015231*pi,0.32992043302514995*pi) q[25];
U1q(0.533284356876129*pi,0.2370722288021403*pi) q[26];
U1q(0.220007907671423*pi,0.8769257991691903*pi) q[27];
U1q(0.366002437344043*pi,1.8805132585120399*pi) q[28];
U1q(0.752420483330941*pi,0.7993053014022706*pi) q[29];
U1q(0.309440478453943*pi,1.6175136329394597*pi) q[30];
U1q(0.591699540979727*pi,1.7329983547846703*pi) q[31];
U1q(0.298092545031466*pi,1.1683862659058608*pi) q[32];
U1q(0.809633822571842*pi,1.4008760948529302*pi) q[33];
U1q(0.576607524772113*pi,0.004532405169170417*pi) q[34];
U1q(0.372781025396515*pi,1.4232474495735694*pi) q[35];
U1q(0.200945898220509*pi,1.4402864052950903*pi) q[36];
U1q(0.716978701326579*pi,0.35682062031488027*pi) q[37];
U1q(0.48967318017577*pi,0.2771631479704997*pi) q[38];
U1q(0.667952585378373*pi,1.5000104562166099*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[2],q[22];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[27],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[15];
RZZ(0.5*pi) q[36],q[14];
RZZ(0.5*pi) q[20],q[17];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[32],q[19];
RZZ(0.5*pi) q[21],q[23];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[35],q[26];
RZZ(0.5*pi) q[29],q[28];
U1q(0.328538065891459*pi,1.3583098196927104*pi) q[0];
U1q(0.609301831946261*pi,0.7679955932827003*pi) q[1];
U1q(0.557806727910488*pi,0.009822461347240186*pi) q[2];
U1q(0.701011283484626*pi,0.6107552031029009*pi) q[3];
U1q(0.684730787501643*pi,0.9027207086211106*pi) q[4];
U1q(0.143798028548841*pi,1.1455556990608908*pi) q[5];
U1q(0.747378799485378*pi,0.15109677926307974*pi) q[6];
U1q(0.528975095210992*pi,1.4817144464745002*pi) q[7];
U1q(0.449133087677523*pi,1.7841005861897603*pi) q[8];
U1q(0.468683544260424*pi,0.1592772908180704*pi) q[9];
U1q(0.405510524321363*pi,0.6866994279444292*pi) q[10];
U1q(0.414373160033282*pi,1.15223374119505*pi) q[11];
U1q(0.656898191358227*pi,1.3929047185521704*pi) q[12];
U1q(0.807485244285372*pi,1.2105338574003195*pi) q[13];
U1q(0.377963590663508*pi,0.3802669730370498*pi) q[14];
U1q(0.550479641162911*pi,1.4730261877656492*pi) q[15];
U1q(0.440344220586073*pi,0.4202570420423797*pi) q[16];
U1q(0.179699262277664*pi,0.9310684024564395*pi) q[17];
U1q(0.78667780372147*pi,0.8339295041619703*pi) q[18];
U1q(0.585695049228224*pi,0.4525578477057195*pi) q[19];
U1q(0.642002813268734*pi,1.3259775547294002*pi) q[20];
U1q(0.374567506425764*pi,1.5159842770474992*pi) q[21];
U1q(0.412455263224461*pi,0.19699117828440027*pi) q[22];
U1q(0.874484247402051*pi,0.32419615907561017*pi) q[23];
U1q(0.744912527264995*pi,0.7914851188584109*pi) q[24];
U1q(0.271951510079084*pi,0.27195359677261965*pi) q[25];
U1q(0.857307801953017*pi,0.7744544815322403*pi) q[26];
U1q(0.335916469226083*pi,0.24843232774360047*pi) q[27];
U1q(0.610353595490024*pi,0.6270845026372207*pi) q[28];
U1q(0.647122274754817*pi,0.7303975005380998*pi) q[29];
U1q(0.482329030051973*pi,1.0790861220318995*pi) q[30];
U1q(0.119740532770582*pi,0.9761119221773793*pi) q[31];
U1q(0.49070260171391*pi,0.3836788012871999*pi) q[32];
U1q(0.479405298388864*pi,0.07710045929759968*pi) q[33];
U1q(0.551042019761886*pi,1.3067408608930808*pi) q[34];
U1q(0.604737882639691*pi,1.9886575702961*pi) q[35];
U1q(0.689216059508752*pi,1.9882348865923998*pi) q[36];
U1q(0.484365337953779*pi,0.004916743735300599*pi) q[37];
U1q(0.436842447521505*pi,1.1112747724294003*pi) q[38];
U1q(0.599788533558892*pi,0.7104056033250696*pi) q[39];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[36],q[4];
RZZ(0.5*pi) q[13],q[5];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[7],q[30];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[12],q[27];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[17],q[21];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[29];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[33],q[31];
RZZ(0.5*pi) q[37],q[35];
RZZ(0.5*pi) q[39],q[38];
U1q(0.86936954826955*pi,1.5585632339018094*pi) q[0];
U1q(0.413278175751064*pi,0.34218998548310076*pi) q[1];
U1q(0.32377754224581*pi,0.7539945338912997*pi) q[2];
U1q(0.769157028171699*pi,1.3686109878702002*pi) q[3];
U1q(0.339814638091902*pi,0.6825192205500006*pi) q[4];
U1q(0.527471946722549*pi,0.8484684500072994*pi) q[5];
U1q(0.524822477708161*pi,1.2803170133690998*pi) q[6];
U1q(0.436216963973003*pi,1.7186540694808006*pi) q[7];
U1q(0.799432479848092*pi,0.35483205103521964*pi) q[8];
U1q(0.397133799486792*pi,1.5598444862619*pi) q[9];
U1q(0.36735245756347*pi,1.8396398488042998*pi) q[10];
U1q(0.652000249556417*pi,0.16251326457171*pi) q[11];
U1q(0.549308012478864*pi,1.1984131118938404*pi) q[12];
U1q(0.440662607879144*pi,1.5889802306058005*pi) q[13];
U1q(0.562242969847244*pi,1.6034527178778006*pi) q[14];
U1q(0.397229365136665*pi,0.1434375301223607*pi) q[15];
U1q(0.396918123571687*pi,0.15273429203710087*pi) q[16];
U1q(0.531321852374482*pi,0.8014460894171993*pi) q[17];
U1q(0.429257338545769*pi,0.3637622480186007*pi) q[18];
U1q(0.571790191037729*pi,0.00946323676829941*pi) q[19];
U1q(0.758655729077704*pi,0.7123719664361996*pi) q[20];
U1q(0.663144498129812*pi,1.0450477297753*pi) q[21];
U1q(0.226567783887582*pi,0.10715837330096978*pi) q[22];
U1q(0.513681261314426*pi,1.6294362938441296*pi) q[23];
U1q(0.555952835873775*pi,1.5355937713787*pi) q[24];
U1q(0.449254345656685*pi,1.6604295746235191*pi) q[25];
U1q(0.438791730454018*pi,0.4555715697054996*pi) q[26];
U1q(0.693678617188932*pi,1.6279533186284993*pi) q[27];
U1q(0.384425437040003*pi,0.3519694240609006*pi) q[28];
U1q(0.796621389386434*pi,0.7580939653970002*pi) q[29];
U1q(0.655897007514867*pi,0.5984303477350004*pi) q[30];
U1q(0.381359188127689*pi,1.4893527594029*pi) q[31];
U1q(0.240140271591878*pi,0.9808715474359992*pi) q[32];
U1q(0.404382302240184*pi,0.6661040197003398*pi) q[33];
U1q(0.500797585992677*pi,0.7038647618729996*pi) q[34];
U1q(0.534792594098704*pi,0.13898126857909965*pi) q[35];
U1q(0.913339172074449*pi,0.8829501475401003*pi) q[36];
U1q(0.395694984935534*pi,1.2428776467001992*pi) q[37];
U1q(0.362754068516046*pi,0.7227673959194991*pi) q[38];
U1q(0.365051882591339*pi,1.7931501198911004*pi) q[39];
RZZ(0.5*pi) q[0],q[26];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[9],q[31];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[38],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[27],q[22];
RZZ(0.5*pi) q[39],q[29];
RZZ(0.5*pi) q[37],q[36];
U1q(0.616562718649063*pi,0.5940360542698997*pi) q[0];
U1q(0.861294162704822*pi,1.9390024398909986*pi) q[1];
U1q(0.281618970667745*pi,0.6988576135961999*pi) q[2];
U1q(0.131623493858377*pi,1.630094750356399*pi) q[3];
U1q(0.325431595356915*pi,0.8480838006984008*pi) q[4];
U1q(0.335940442381092*pi,0.9877612502731008*pi) q[5];
U1q(0.911173580220646*pi,1.0743010377337008*pi) q[6];
U1q(0.68464912280992*pi,0.41418061460790057*pi) q[7];
U1q(0.487158745758586*pi,1.2944789151997007*pi) q[8];
U1q(0.594368644557523*pi,1.8637664219152992*pi) q[9];
U1q(0.0297832504095081*pi,1.5757827679476009*pi) q[10];
U1q(0.431255426896072*pi,0.39547275125578985*pi) q[11];
U1q(0.630473893122422*pi,0.43124093483212*pi) q[12];
U1q(0.388289001407162*pi,1.8221711773325993*pi) q[13];
U1q(0.336436308644942*pi,0.7141027977732008*pi) q[14];
U1q(0.117393834899392*pi,1.7285219389040005*pi) q[15];
U1q(0.255480989279196*pi,1.096354378079301*pi) q[16];
U1q(0.946404435337754*pi,0.11764691884440026*pi) q[17];
U1q(0.51774642515002*pi,1.5444889054664994*pi) q[18];
U1q(0.303405326850798*pi,1.1220760876528004*pi) q[19];
U1q(0.420118188524465*pi,1.3247577603178016*pi) q[20];
U1q(0.797516174809515*pi,0.9943461691668993*pi) q[21];
U1q(0.0978559059993506*pi,1.7142907619760006*pi) q[22];
U1q(0.554102778587501*pi,1.1554724457763008*pi) q[23];
U1q(0.438365950861776*pi,0.6624086974006005*pi) q[24];
U1q(0.0764902896551922*pi,1.3624175146108701*pi) q[25];
U1q(0.604846710798791*pi,1.5664913065013994*pi) q[26];
U1q(0.694750368017601*pi,1.5013146420866992*pi) q[27];
U1q(0.196362930524847*pi,1.7762845014897*pi) q[28];
U1q(0.611559054979941*pi,1.4720275296498002*pi) q[29];
U1q(0.722726623349927*pi,0.7835374789154006*pi) q[30];
U1q(0.415821896348488*pi,1.1426177752311002*pi) q[31];
U1q(0.824373549133773*pi,1.8634649547233018*pi) q[32];
U1q(0.779075417172573*pi,0.6978918068375002*pi) q[33];
U1q(0.185048411535089*pi,1.8449250033833007*pi) q[34];
U1q(0.901687992350222*pi,0.09148747400289992*pi) q[35];
U1q(0.360230343208617*pi,0.6109055955877007*pi) q[36];
U1q(0.518265928978659*pi,1.4190028073106014*pi) q[37];
U1q(0.639130062337492*pi,1.0543086244932987*pi) q[38];
U1q(0.239531017742654*pi,0.9715570351312994*pi) q[39];
RZZ(0.5*pi) q[35],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[28];
RZZ(0.5*pi) q[37],q[4];
RZZ(0.5*pi) q[5],q[18];
RZZ(0.5*pi) q[39],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[36],q[11];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[16],q[34];
RZZ(0.5*pi) q[19],q[22];
RZZ(0.5*pi) q[21],q[26];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[32],q[25];
RZZ(0.5*pi) q[33],q[29];
RZZ(0.5*pi) q[38],q[30];
U1q(0.553317945525664*pi,1.1237568254839996*pi) q[0];
U1q(0.290941433781721*pi,0.7186381175731995*pi) q[1];
U1q(0.410595024993571*pi,0.1744598823760004*pi) q[2];
U1q(0.697542669546724*pi,0.8803192870651984*pi) q[3];
U1q(0.182139249535777*pi,0.35568773661020003*pi) q[4];
U1q(0.636959086185795*pi,0.9411794054745997*pi) q[5];
U1q(0.272133362299962*pi,1.1430893341839994*pi) q[6];
U1q(0.589182429537112*pi,1.0829165361991002*pi) q[7];
U1q(0.564511929735176*pi,0.30837221954960015*pi) q[8];
U1q(0.496356995301713*pi,1.5925344868816005*pi) q[9];
U1q(0.598914532422463*pi,0.8635525625356983*pi) q[10];
U1q(0.719907722338642*pi,1.3483115474650997*pi) q[11];
U1q(0.786559321629865*pi,1.9734665892225003*pi) q[12];
U1q(0.538305963841311*pi,0.03745977084810015*pi) q[13];
U1q(0.758448001587858*pi,0.6425701836689992*pi) q[14];
U1q(0.596426652977904*pi,1.9441051185662985*pi) q[15];
U1q(0.420518392060615*pi,1.3065052985880001*pi) q[16];
U1q(0.587959491452266*pi,1.0600618458793996*pi) q[17];
U1q(0.718260051410775*pi,1.6522226566181004*pi) q[18];
U1q(0.499502053958136*pi,0.6729616114415009*pi) q[19];
U1q(0.113283514156267*pi,0.08256294522229979*pi) q[20];
U1q(0.667839685292054*pi,1.7679075016429984*pi) q[21];
U1q(0.820929325480545*pi,0.4393899846070006*pi) q[22];
U1q(0.739207886940038*pi,0.9158212410695992*pi) q[23];
U1q(0.346756100216822*pi,0.7877817333422001*pi) q[24];
U1q(0.653428208686791*pi,1.6803098319293994*pi) q[25];
U1q(0.517696540327821*pi,0.4684428642916991*pi) q[26];
U1q(0.570285831925438*pi,1.9402249991015985*pi) q[27];
U1q(0.912132945164806*pi,0.4093525334654995*pi) q[28];
U1q(0.0713768357980965*pi,0.2101690389799007*pi) q[29];
U1q(0.485764524344525*pi,1.0005510203535017*pi) q[30];
U1q(0.327634248815335*pi,1.4206456259706997*pi) q[31];
U1q(0.311586922830077*pi,1.7161218473846986*pi) q[32];
U1q(0.343728994618107*pi,0.12784696482719937*pi) q[33];
U1q(0.69191812225676*pi,1.8918224371605987*pi) q[34];
U1q(0.626527082529425*pi,0.7942785680195001*pi) q[35];
U1q(0.623928892495141*pi,0.43581864904849965*pi) q[36];
U1q(0.42393287659497*pi,0.8158441282267006*pi) q[37];
U1q(0.623890397300667*pi,1.4033360874862986*pi) q[38];
U1q(0.34926862819076*pi,0.7257070670310988*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[9];
RZZ(0.5*pi) q[34],q[3];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[6],q[11];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[10],q[36];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[31];
RZZ(0.5*pi) q[32],q[15];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[37],q[18];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[39],q[21];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[27],q[26];
RZZ(0.5*pi) q[30],q[29];
U1q(0.70985166636463*pi,1.8408892022972996*pi) q[0];
U1q(0.468055687163859*pi,1.4320782672042007*pi) q[1];
U1q(0.425479562970858*pi,0.7029638098989004*pi) q[2];
U1q(0.537442407096894*pi,1.3096265298120997*pi) q[3];
U1q(0.635630822386104*pi,1.2725884120299007*pi) q[4];
U1q(0.674122548035963*pi,1.8708095279039014*pi) q[5];
U1q(0.313690247521155*pi,1.8388694896821*pi) q[6];
U1q(0.360912742041974*pi,0.6785870159985983*pi) q[7];
U1q(0.508623411921986*pi,1.6390875155382005*pi) q[8];
U1q(0.0942162370602745*pi,0.6209396342181996*pi) q[9];
U1q(0.423784551611801*pi,0.27794047121729903*pi) q[10];
U1q(0.300989016955645*pi,1.4605447974791996*pi) q[11];
U1q(0.493901663058813*pi,0.4366919061789005*pi) q[12];
U1q(0.925811279229706*pi,0.035775764559199175*pi) q[13];
U1q(0.241038591550353*pi,0.5447573812209008*pi) q[14];
U1q(0.431797715149713*pi,0.5974478635799017*pi) q[15];
U1q(0.354513131126301*pi,1.0780038421615004*pi) q[16];
U1q(0.398062061179874*pi,0.6332766201944011*pi) q[17];
U1q(0.571606983808291*pi,1.3029413823559999*pi) q[18];
U1q(0.269854922366213*pi,1.4944855812735014*pi) q[19];
U1q(0.779330603209004*pi,0.342149740317101*pi) q[20];
U1q(0.338308110361672*pi,1.6451329984065985*pi) q[21];
U1q(0.805699688719449*pi,0.4518553462766004*pi) q[22];
U1q(0.557185545023583*pi,0.03047020462970096*pi) q[23];
U1q(0.663651574941028*pi,0.9020828272511991*pi) q[24];
U1q(0.124997103812152*pi,1.0235901021203997*pi) q[25];
U1q(0.805696079280656*pi,1.2215868827318985*pi) q[26];
U1q(0.311705185808243*pi,1.9905340927444009*pi) q[27];
U1q(0.399845068234921*pi,1.2714341041205017*pi) q[28];
U1q(0.592914471265182*pi,1.4833275566359987*pi) q[29];
U1q(0.78078347257787*pi,0.5207139144770991*pi) q[30];
U1q(0.807698671898539*pi,1.442977585091299*pi) q[31];
U1q(0.452599339496034*pi,1.3069260485577985*pi) q[32];
U1q(0.833017867707446*pi,1.4500858511269996*pi) q[33];
U1q(0.544908321008563*pi,0.005405921702699601*pi) q[34];
U1q(0.63224831039675*pi,0.5203363807694998*pi) q[35];
U1q(0.7108833224634*pi,0.9770036848959016*pi) q[36];
U1q(0.391891775740576*pi,0.8236828461382011*pi) q[37];
U1q(0.268324029349614*pi,0.7273165459483018*pi) q[38];
U1q(0.558711587065429*pi,1.0268956433931002*pi) q[39];
rz(1.4261057867685007*pi) q[0];
rz(2.8880763256244*pi) q[1];
rz(2.059510406905801*pi) q[2];
rz(3.369796506425299*pi) q[3];
rz(0.11654554476070089*pi) q[4];
rz(3.421267269531601*pi) q[5];
rz(2.8271367340432008*pi) q[6];
rz(1.5119990597027986*pi) q[7];
rz(2.6919839501272005*pi) q[8];
rz(2.893350605374401*pi) q[9];
rz(0.6628349880915003*pi) q[10];
rz(1.6551869003892001*pi) q[11];
rz(2.5422846729243*pi) q[12];
rz(0.4225041845238984*pi) q[13];
rz(3.7873766936604003*pi) q[14];
rz(3.2976440473967017*pi) q[15];
rz(2.3337171071368985*pi) q[16];
rz(2.1175422522277003*pi) q[17];
rz(1.750921543006399*pi) q[18];
rz(3.5632482926431983*pi) q[19];
rz(3.7391188035952005*pi) q[20];
rz(1.4531542801839983*pi) q[21];
rz(3.995248978841399*pi) q[22];
rz(2.103246890165501*pi) q[23];
rz(2.8212817900859015*pi) q[24];
rz(0.5395036999706999*pi) q[25];
rz(1.370726347135399*pi) q[26];
rz(2.8274739999660987*pi) q[27];
rz(3.3004132237133987*pi) q[28];
rz(3.338011942262*pi) q[29];
rz(0.3966467236492015*pi) q[30];
rz(1.2393667263878*pi) q[31];
rz(2.094407842325399*pi) q[32];
rz(1.0972374346431017*pi) q[33];
rz(3.2456710785941*pi) q[34];
rz(0.2952718893773003*pi) q[35];
rz(0.29422807587259925*pi) q[36];
rz(1.2832839026943006*pi) q[37];
rz(1.8416612628755011*pi) q[38];
rz(2.7998416445924015*pi) q[39];
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
