OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.323424972798274*pi,0.569990574847*pi) q[0];
U1q(0.117020724698537*pi,1.108829679916906*pi) q[1];
U1q(0.369720430616806*pi,0.9523126507644899*pi) q[2];
U1q(0.0650515824805898*pi,0.8520715240024499*pi) q[3];
U1q(0.451365041936077*pi,0.340819760918027*pi) q[4];
U1q(0.322922915545349*pi,1.77555853796206*pi) q[5];
U1q(0.659894339734807*pi,0.870566398021417*pi) q[6];
U1q(0.794504461244371*pi,1.39557597933374*pi) q[7];
U1q(0.310556070023406*pi,0.392515108009026*pi) q[8];
U1q(0.725650481475715*pi,1.5842219197023*pi) q[9];
U1q(0.454292341537507*pi,0.381043036404625*pi) q[10];
U1q(0.337374638002845*pi,0.5626551349359801*pi) q[11];
U1q(0.960402188492591*pi,0.507451344197499*pi) q[12];
U1q(0.754171853860141*pi,1.9450855001788765*pi) q[13];
U1q(0.548640496999582*pi,0.253064477319722*pi) q[14];
U1q(0.63111537447641*pi,0.37613905807844*pi) q[15];
U1q(0.531518043246139*pi,1.0448964770457772*pi) q[16];
U1q(0.652952092344381*pi,0.0531061724357022*pi) q[17];
U1q(0.343184292833215*pi,1.03535363557764*pi) q[18];
U1q(0.486217439230268*pi,0.8498076304193101*pi) q[19];
U1q(0.815752334632645*pi,0.86591054110341*pi) q[20];
U1q(0.479424951012567*pi,0.321445560367774*pi) q[21];
U1q(0.582248215807692*pi,0.183157516505851*pi) q[22];
U1q(0.343895219344201*pi,0.133977030353274*pi) q[23];
U1q(0.508042964819232*pi,0.857354652004601*pi) q[24];
U1q(0.850945625500195*pi,0.0611885328480853*pi) q[25];
U1q(0.621183081850427*pi,0.612465266844911*pi) q[26];
U1q(0.16021250295271*pi,1.195995698514049*pi) q[27];
U1q(0.371494685430046*pi,1.569611922985641*pi) q[28];
U1q(0.913532657054813*pi,0.559852132755245*pi) q[29];
U1q(0.169086783867265*pi,1.172513776135266*pi) q[30];
U1q(0.45268667215591*pi,0.90308850620361*pi) q[31];
U1q(0.845825964544879*pi,1.210139411574029*pi) q[32];
U1q(0.396494641028094*pi,0.8707517299762499*pi) q[33];
U1q(0.217206586451144*pi,1.543099690806451*pi) q[34];
U1q(0.356170997228015*pi,0.6965420024401701*pi) q[35];
U1q(0.849683821888839*pi,1.188785965823942*pi) q[36];
U1q(0.300747400707877*pi,0.56988870916383*pi) q[37];
U1q(0.816974996026752*pi,0.688423596063248*pi) q[38];
U1q(0.82566925826418*pi,0.862089019697056*pi) q[39];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[2],q[34];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[20],q[10];
RZZ(0.5*pi) q[30],q[11];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[24],q[31];
RZZ(0.5*pi) q[38],q[28];
RZZ(0.5*pi) q[39],q[36];
U1q(0.637072081417699*pi,1.29533551007952*pi) q[0];
U1q(0.313473539665032*pi,0.5044532933491799*pi) q[1];
U1q(0.416705194206493*pi,1.0427761022016102*pi) q[2];
U1q(0.364168463087964*pi,0.9172532823120401*pi) q[3];
U1q(0.948759185140785*pi,1.5255626747108102*pi) q[4];
U1q(0.257137920336542*pi,1.57812008019364*pi) q[5];
U1q(0.664891593346356*pi,0.8802702822477*pi) q[6];
U1q(0.346382659964823*pi,0.8981976191830701*pi) q[7];
U1q(0.83615400118978*pi,1.1470248516018802*pi) q[8];
U1q(0.672208993805964*pi,0.98056804673862*pi) q[9];
U1q(0.587163332386159*pi,1.9356580355127*pi) q[10];
U1q(0.45138185961423*pi,1.78203948855292*pi) q[11];
U1q(0.874248627591448*pi,1.73837779264466*pi) q[12];
U1q(0.561378860472484*pi,0.5936672513071899*pi) q[13];
U1q(0.740876196881305*pi,1.2599769626926198*pi) q[14];
U1q(0.459500446598796*pi,0.9716436857284698*pi) q[15];
U1q(0.154483497089868*pi,0.6020492049659101*pi) q[16];
U1q(0.298173812120911*pi,0.05057682130243002*pi) q[17];
U1q(0.28978558114939*pi,1.0377036454530901*pi) q[18];
U1q(0.32891211484045*pi,0.7595023728665402*pi) q[19];
U1q(0.860623926393698*pi,1.2355339625871098*pi) q[20];
U1q(0.276989785969006*pi,1.93014370067662*pi) q[21];
U1q(0.498349360917882*pi,0.59263702801768*pi) q[22];
U1q(0.414969050372559*pi,0.7670519598141499*pi) q[23];
U1q(0.948555859692527*pi,1.392959688746958*pi) q[24];
U1q(0.666428632385483*pi,1.34340162193745*pi) q[25];
U1q(0.771415790613315*pi,1.4286550183022002*pi) q[26];
U1q(0.536893152825783*pi,1.3018088847265599*pi) q[27];
U1q(0.695374404995059*pi,0.14028409668962993*pi) q[28];
U1q(0.715436076367441*pi,0.8344317902871699*pi) q[29];
U1q(0.501606341566246*pi,0.6180645969218499*pi) q[30];
U1q(0.269500607808447*pi,1.55366869136203*pi) q[31];
U1q(0.49241939625117*pi,1.8012680215819898*pi) q[32];
U1q(0.346264598516483*pi,1.46409579158929*pi) q[33];
U1q(0.619399945098066*pi,1.3159673738042401*pi) q[34];
U1q(0.235060887955386*pi,0.6769433793576201*pi) q[35];
U1q(0.66401066909607*pi,1.6074804304516501*pi) q[36];
U1q(0.745613146533357*pi,0.030371119090550014*pi) q[37];
U1q(0.567565456782401*pi,1.421636870019493*pi) q[38];
U1q(0.226050751838293*pi,0.94265400436786*pi) q[39];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[32];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[33],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[37];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[12],q[31];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[30],q[19];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[26],q[25];
RZZ(0.5*pi) q[27],q[34];
U1q(0.755061431444844*pi,0.64069805251287*pi) q[0];
U1q(0.313764161013538*pi,0.3723633061860703*pi) q[1];
U1q(0.0741638503205734*pi,0.8982770957681696*pi) q[2];
U1q(0.701328401614442*pi,1.9936368918306098*pi) q[3];
U1q(0.284799382298937*pi,0.61880123584627*pi) q[4];
U1q(0.613335782772091*pi,1.18894311334234*pi) q[5];
U1q(0.460111685764927*pi,1.3154904333750101*pi) q[6];
U1q(0.692169778357368*pi,0.9232667238972398*pi) q[7];
U1q(0.417880400860195*pi,0.69140187775138*pi) q[8];
U1q(0.673980152316459*pi,0.07488831796081019*pi) q[9];
U1q(0.547607069223587*pi,1.2496782501266903*pi) q[10];
U1q(0.106310794100475*pi,1.39643170727659*pi) q[11];
U1q(0.758655654249436*pi,1.6635457217871803*pi) q[12];
U1q(0.682571358200184*pi,0.8634104486699301*pi) q[13];
U1q(0.711150463472486*pi,0.04133348121129998*pi) q[14];
U1q(0.346674022683983*pi,0.40747340830526024*pi) q[15];
U1q(0.395515266042887*pi,0.39635547124405957*pi) q[16];
U1q(0.448083240131666*pi,0.6408099462078596*pi) q[17];
U1q(0.282942521965793*pi,1.8338601828055197*pi) q[18];
U1q(0.396507979487757*pi,0.4955565330344198*pi) q[19];
U1q(0.525738959931571*pi,1.9932906187207102*pi) q[20];
U1q(0.86478129583615*pi,0.015056789842589957*pi) q[21];
U1q(0.765846377075963*pi,0.46893874514212985*pi) q[22];
U1q(0.63843440449193*pi,0.2755683291761004*pi) q[23];
U1q(0.365755134065694*pi,1.75085938523736*pi) q[24];
U1q(0.776620349315957*pi,0.6945207839801997*pi) q[25];
U1q(0.920206468454683*pi,0.1944890196769702*pi) q[26];
U1q(0.545954193937178*pi,0.16748600515385004*pi) q[27];
U1q(0.642236323955405*pi,1.45395187997262*pi) q[28];
U1q(0.39293376271059*pi,1.0339552698602201*pi) q[29];
U1q(0.62941682137329*pi,1.76533900042137*pi) q[30];
U1q(0.183648683629206*pi,1.21471144338767*pi) q[31];
U1q(0.34954286985725*pi,1.35291482531761*pi) q[32];
U1q(0.603603218139446*pi,1.85947438646628*pi) q[33];
U1q(0.462886357395842*pi,0.6721669374850698*pi) q[34];
U1q(0.255746863432629*pi,1.1592725628589804*pi) q[35];
U1q(0.178802743349367*pi,1.82771766664054*pi) q[36];
U1q(0.634360953859342*pi,1.5081857203945797*pi) q[37];
U1q(0.635223933939456*pi,1.51804802837446*pi) q[38];
U1q(0.704652688439545*pi,0.9507636687797998*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[39],q[2];
RZZ(0.5*pi) q[29],q[3];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[26],q[9];
RZZ(0.5*pi) q[33],q[10];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[13],q[16];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[15],q[32];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[23],q[37];
RZZ(0.5*pi) q[27],q[31];
RZZ(0.5*pi) q[38],q[34];
U1q(0.0879498445183793*pi,0.9149822926743099*pi) q[0];
U1q(0.640761251847513*pi,1.5543570617093199*pi) q[1];
U1q(0.173224628777469*pi,1.1871005697229595*pi) q[2];
U1q(0.47801640989414*pi,0.7299580357381998*pi) q[3];
U1q(0.581359220938008*pi,1.8892860221285908*pi) q[4];
U1q(0.230758051560616*pi,1.9820356337293203*pi) q[5];
U1q(0.182476332609027*pi,1.0430486996014796*pi) q[6];
U1q(0.548846406118853*pi,0.32551638677943995*pi) q[7];
U1q(0.532354874318849*pi,1.1936132908039099*pi) q[8];
U1q(0.345756279713684*pi,0.5867396650766601*pi) q[9];
U1q(0.388711933538486*pi,1.4379343766885704*pi) q[10];
U1q(0.72757145225228*pi,1.7948138582834101*pi) q[11];
U1q(0.334761843811347*pi,1.4681325879406097*pi) q[12];
U1q(0.866327683421816*pi,1.64335357453486*pi) q[13];
U1q(0.758863823333195*pi,0.19544361087785056*pi) q[14];
U1q(0.476554247943702*pi,0.8472239413515901*pi) q[15];
U1q(0.824501531807297*pi,1.9921051356495099*pi) q[16];
U1q(0.217103643558048*pi,1.4146224359349695*pi) q[17];
U1q(0.449381138284813*pi,0.24564087774998988*pi) q[18];
U1q(0.81989583713751*pi,1.2194071817576901*pi) q[19];
U1q(0.65296726035512*pi,1.8065441710460401*pi) q[20];
U1q(0.663738656658899*pi,1.1751251788445796*pi) q[21];
U1q(0.798432141306052*pi,1.24040980096325*pi) q[22];
U1q(0.340186490938106*pi,0.07043620947593965*pi) q[23];
U1q(0.859335708804304*pi,0.20430662087444995*pi) q[24];
U1q(0.406047778972723*pi,0.5093713108876106*pi) q[25];
U1q(0.932802159148945*pi,1.4065870651609798*pi) q[26];
U1q(0.237383966510998*pi,1.5077505876016097*pi) q[27];
U1q(0.768597208585327*pi,1.44238995569822*pi) q[28];
U1q(0.773601239104538*pi,1.0739640725807007*pi) q[29];
U1q(0.605602844999779*pi,0.6852293089697601*pi) q[30];
U1q(0.528558205165463*pi,1.2758320097044393*pi) q[31];
U1q(0.729251326444652*pi,0.5745378280859903*pi) q[32];
U1q(0.57484530143404*pi,0.58577555916919*pi) q[33];
U1q(0.298824111219562*pi,1.7815284544037393*pi) q[34];
U1q(0.267167209771156*pi,1.9242026246794008*pi) q[35];
U1q(0.773359921821825*pi,1.7396188793739*pi) q[36];
U1q(0.696454168338003*pi,0.37504705634241997*pi) q[37];
U1q(0.679645665945043*pi,0.9719277162095699*pi) q[38];
U1q(0.0867554567381706*pi,1.7477191373113499*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[2],q[35];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[38],q[4];
RZZ(0.5*pi) q[26],q[5];
RZZ(0.5*pi) q[25],q[7];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[22],q[15];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[27],q[18];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[20],q[31];
RZZ(0.5*pi) q[24],q[32];
U1q(0.527235993806521*pi,1.8209555489107991*pi) q[0];
U1q(0.3590868281356*pi,1.9132407757799292*pi) q[1];
U1q(0.763875166377195*pi,1.0686983512831993*pi) q[2];
U1q(0.863282920223718*pi,0.6189944174186497*pi) q[3];
U1q(0.244478409897347*pi,1.9006107164146009*pi) q[4];
U1q(0.281649098299477*pi,1.2229950042972497*pi) q[5];
U1q(0.605829728479295*pi,0.40757794924337*pi) q[6];
U1q(0.244709387315843*pi,0.1867200159985103*pi) q[7];
U1q(0.863857011661293*pi,1.0442405881727703*pi) q[8];
U1q(0.785525133175147*pi,0.07976196024802995*pi) q[9];
U1q(0.380734954914187*pi,0.5943465726012995*pi) q[10];
U1q(0.289560886763299*pi,0.7210016307811102*pi) q[11];
U1q(0.780400502275218*pi,1.5406891103332203*pi) q[12];
U1q(0.5639668251193*pi,1.2371102889797099*pi) q[13];
U1q(0.671352267267348*pi,0.7793588367783997*pi) q[14];
U1q(0.561976748113353*pi,0.06975763825058934*pi) q[15];
U1q(0.268969906393481*pi,1.1320140753986205*pi) q[16];
U1q(0.253629454669271*pi,0.1442842745709303*pi) q[17];
U1q(0.331054547957528*pi,0.9072206512362104*pi) q[18];
U1q(0.887440628387704*pi,1.3736330664891003*pi) q[19];
U1q(0.434828105212461*pi,1.0682402491918204*pi) q[20];
U1q(0.781923912145384*pi,1.2404067474471496*pi) q[21];
U1q(0.269614561681436*pi,0.9708894652476001*pi) q[22];
U1q(0.760961569982412*pi,1.0308237296786995*pi) q[23];
U1q(0.65717985748736*pi,1.8312972429054*pi) q[24];
U1q(0.756003127246174*pi,1.1416463638496008*pi) q[25];
U1q(0.773832055335955*pi,0.68452846303855*pi) q[26];
U1q(0.370448748868624*pi,1.3644626915962998*pi) q[27];
U1q(0.442438018416687*pi,0.17645806840870026*pi) q[28];
U1q(0.415555338071832*pi,0.8483048935406003*pi) q[29];
U1q(0.911194793640248*pi,0.40104196599863995*pi) q[30];
U1q(0.90490109844402*pi,0.32348499293219923*pi) q[31];
U1q(0.427741275181094*pi,0.9894575373942001*pi) q[32];
U1q(0.753679996986989*pi,0.3308152278568599*pi) q[33];
U1q(0.618759283569295*pi,1.5529571084192995*pi) q[34];
U1q(0.924100338266868*pi,1.9435870128467005*pi) q[35];
U1q(0.751484757427566*pi,0.056804437277099495*pi) q[36];
U1q(0.305834517748448*pi,0.19579541378305976*pi) q[37];
U1q(0.803245329551792*pi,0.8130816391480602*pi) q[38];
U1q(0.717548972257482*pi,1.37130375434048*pi) q[39];
RZZ(0.5*pi) q[38],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[27],q[3];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[6],q[30];
RZZ(0.5*pi) q[7],q[11];
RZZ(0.5*pi) q[23],q[8];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[13],q[12];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[16],q[24];
RZZ(0.5*pi) q[39],q[17];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[22],q[19];
RZZ(0.5*pi) q[35],q[31];
RZZ(0.5*pi) q[32],q[36];
RZZ(0.5*pi) q[33],q[34];
U1q(0.900650355238003*pi,1.9461657900888998*pi) q[0];
U1q(0.371258515730507*pi,0.5681784856883993*pi) q[1];
U1q(0.416884840618399*pi,1.0347977450831003*pi) q[2];
U1q(0.501032891373505*pi,1.0205655603752*pi) q[3];
U1q(0.730650640776353*pi,0.4620507972718997*pi) q[4];
U1q(0.68379756285864*pi,1.6719618823522104*pi) q[5];
U1q(0.505371259150754*pi,1.8707375606227004*pi) q[6];
U1q(0.899195422126831*pi,1.0381500605704996*pi) q[7];
U1q(0.476409298096976*pi,0.5292085034206604*pi) q[8];
U1q(0.425838814167937*pi,1.4843567003381999*pi) q[9];
U1q(0.474723576353369*pi,1.9945427567338996*pi) q[10];
U1q(0.847917155074956*pi,0.8561505794939404*pi) q[11];
U1q(0.520789803113529*pi,1.5827944802753997*pi) q[12];
U1q(0.549185211972763*pi,0.01637555786309086*pi) q[13];
U1q(0.511998019849642*pi,0.3629969236212993*pi) q[14];
U1q(0.0568688046785321*pi,0.13470430890519935*pi) q[15];
U1q(0.636157999908409*pi,0.11212526596539973*pi) q[16];
U1q(0.611759326124646*pi,0.3286559014270001*pi) q[17];
U1q(0.951209719186729*pi,0.06581700430690063*pi) q[18];
U1q(0.0211501893399863*pi,1.9349933233767995*pi) q[19];
U1q(0.118386764293691*pi,1.1436950688128*pi) q[20];
U1q(0.735960292359554*pi,0.28047088923620933*pi) q[21];
U1q(0.889924245297919*pi,1.3832729339178993*pi) q[22];
U1q(0.408871835898539*pi,1.0732772545106002*pi) q[23];
U1q(0.843413570154564*pi,0.7725034562371702*pi) q[24];
U1q(0.554175715133662*pi,0.9242262832524002*pi) q[25];
U1q(0.458690349203073*pi,1.7872778327746008*pi) q[26];
U1q(0.740412065735328*pi,1.9694526724972992*pi) q[27];
U1q(0.562629179998552*pi,1.1977084526521802*pi) q[28];
U1q(0.666713490329905*pi,0.43184487537070027*pi) q[29];
U1q(0.30635232985011*pi,0.7405896069029998*pi) q[30];
U1q(0.423479280809721*pi,1.1883882222207003*pi) q[31];
U1q(0.424518693552824*pi,1.4343846282247004*pi) q[32];
U1q(0.742988906136549*pi,1.6151516913443*pi) q[33];
U1q(0.461924683676119*pi,1.2453890915613997*pi) q[34];
U1q(0.900462650549449*pi,0.7094953160989999*pi) q[35];
U1q(0.737769227509361*pi,1.4775943442705*pi) q[36];
U1q(0.312052618181471*pi,1.0312312269103003*pi) q[37];
U1q(0.866701676919922*pi,1.2848248209129602*pi) q[38];
U1q(0.34975526566155*pi,0.6531379834089002*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[38],q[3];
RZZ(0.5*pi) q[37],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[25],q[6];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[8],q[32];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[30];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[35],q[29];
U1q(0.416857328578574*pi,0.6850107908003995*pi) q[0];
U1q(0.0873466836632361*pi,0.7457656683564995*pi) q[1];
U1q(0.671311494333123*pi,1.3527828922342007*pi) q[2];
U1q(0.534231431345577*pi,0.6004369325751*pi) q[3];
U1q(0.379633092758972*pi,0.809998406805601*pi) q[4];
U1q(0.24718382331193*pi,1.7095928843788002*pi) q[5];
U1q(0.2762818115359*pi,1.3965930153110993*pi) q[6];
U1q(0.262880825127002*pi,0.11480355160649935*pi) q[7];
U1q(0.159919701088734*pi,0.1187911827760999*pi) q[8];
U1q(0.389621483478485*pi,0.19198331121429923*pi) q[9];
U1q(0.35313454776537*pi,0.000908132718800303*pi) q[10];
U1q(0.364408582392479*pi,0.19430670614990042*pi) q[11];
U1q(0.547992391142942*pi,0.4666666189410993*pi) q[12];
U1q(0.547599306135905*pi,0.30970988517010056*pi) q[13];
U1q(0.471963392631297*pi,1.5985587958212015*pi) q[14];
U1q(0.6752087841904*pi,1.3976422302553004*pi) q[15];
U1q(0.153279287212562*pi,0.13658829198499944*pi) q[16];
U1q(0.353520179928225*pi,0.7164622891573007*pi) q[17];
U1q(0.419520764867174*pi,1.7754288453975988*pi) q[18];
U1q(0.57983898362329*pi,0.02631934899619992*pi) q[19];
U1q(0.081815875613294*pi,1.2071443105512998*pi) q[20];
U1q(0.743722427422292*pi,0.1702171948076998*pi) q[21];
U1q(0.952116038879894*pi,0.008033028508700468*pi) q[22];
U1q(0.479023026575477*pi,1.530012517565801*pi) q[23];
U1q(0.256923822587442*pi,0.6765360730583296*pi) q[24];
U1q(0.25134445863072*pi,1.7300075199178018*pi) q[25];
U1q(0.176637344834566*pi,1.342145392893599*pi) q[26];
U1q(0.744754270566443*pi,0.9680176089037005*pi) q[27];
U1q(0.379204382150881*pi,0.14180406228886078*pi) q[28];
U1q(0.385296517794222*pi,0.4303273706403008*pi) q[29];
U1q(0.876594077481245*pi,1.3424771551143504*pi) q[30];
U1q(0.333490205140991*pi,0.6635138966739014*pi) q[31];
U1q(0.634701387138387*pi,1.4017228097286*pi) q[32];
U1q(0.54789088970072*pi,1.2896182076085*pi) q[33];
U1q(0.514225559027336*pi,0.1622074361197008*pi) q[34];
U1q(0.500572433397677*pi,0.6447193202763017*pi) q[35];
U1q(0.899913159909709*pi,1.6391250495310992*pi) q[36];
U1q(0.340307633055608*pi,1.5349671999180998*pi) q[37];
U1q(0.65255009630215*pi,0.2868322067202502*pi) q[38];
U1q(0.107982491398478*pi,0.08995405743310059*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[5];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[17],q[3];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[12],q[25];
RZZ(0.5*pi) q[30],q[14];
RZZ(0.5*pi) q[16],q[32];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[21],q[19];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[35],q[23];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[27],q[28];
RZZ(0.5*pi) q[34],q[36];
RZZ(0.5*pi) q[38],q[37];
U1q(0.780996450318288*pi,0.5826810920602998*pi) q[0];
U1q(0.133487052377726*pi,0.9883463346256001*pi) q[1];
U1q(0.130166191105074*pi,1.8121226061687992*pi) q[2];
U1q(0.781853479148282*pi,0.5658742135829993*pi) q[3];
U1q(0.716495609771409*pi,0.9988514066912018*pi) q[4];
U1q(0.605350869142291*pi,0.0438990004507005*pi) q[5];
U1q(0.875415862632471*pi,0.3665827451205992*pi) q[6];
U1q(0.45377621015632*pi,0.5659988610539006*pi) q[7];
U1q(0.548635798312858*pi,0.2680703520822991*pi) q[8];
U1q(0.505062386376492*pi,1.6483878634700009*pi) q[9];
U1q(0.518157924315178*pi,1.595808154466301*pi) q[10];
U1q(0.679033302770567*pi,0.7628610637822995*pi) q[11];
U1q(0.209292550223153*pi,0.21316737931159935*pi) q[12];
U1q(0.649693952428589*pi,0.9333167665327995*pi) q[13];
U1q(0.376595383395846*pi,0.10545508925979874*pi) q[14];
U1q(0.703033196544214*pi,0.6127987747427994*pi) q[15];
U1q(0.310768447439882*pi,0.7812705874154986*pi) q[16];
U1q(0.553382732560878*pi,1.9825332870506998*pi) q[17];
U1q(0.468864168285621*pi,1.7320502988422*pi) q[18];
U1q(0.39858666997907*pi,0.0034133537689005067*pi) q[19];
U1q(0.19462346040757*pi,0.7987727627062*pi) q[20];
U1q(0.308994310024519*pi,1.0977070647084002*pi) q[21];
U1q(0.619012404629547*pi,0.1048669662018007*pi) q[22];
U1q(0.41178768816308*pi,0.35065868495039965*pi) q[23];
U1q(0.686741929239582*pi,0.06760536785001925*pi) q[24];
U1q(0.323386133291636*pi,0.3810406289893997*pi) q[25];
U1q(0.809045190418378*pi,1.1659486870926017*pi) q[26];
U1q(0.0533361153027058*pi,1.674872518454901*pi) q[27];
U1q(0.348210231062356*pi,1.1192871293502993*pi) q[28];
U1q(0.327636742149351*pi,1.1573346384824994*pi) q[29];
U1q(0.83951800841366*pi,1.5159538095503002*pi) q[30];
U1q(0.589075580995441*pi,0.12304657352419923*pi) q[31];
U1q(0.369723570049424*pi,1.0314024315505002*pi) q[32];
U1q(0.256070380181988*pi,1.7711169524536992*pi) q[33];
U1q(0.755283143228688*pi,0.5602899114310986*pi) q[34];
U1q(0.641851218425668*pi,1.094575114422799*pi) q[35];
U1q(0.26200566814264*pi,0.20717859314400044*pi) q[36];
U1q(0.529982325304869*pi,0.2956129737463993*pi) q[37];
U1q(0.909079710106628*pi,1.1588867843385007*pi) q[38];
U1q(0.303971870214304*pi,1.7582755509376007*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[12],q[4];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[22],q[8];
RZZ(0.5*pi) q[9],q[29];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[13],q[34];
RZZ(0.5*pi) q[15],q[23];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[25],q[17];
RZZ(0.5*pi) q[31],q[18];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[38],q[21];
RZZ(0.5*pi) q[35],q[24];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[39],q[30];
U1q(0.295408316115514*pi,0.008679572865400331*pi) q[0];
U1q(0.753877937840638*pi,0.1644315901145994*pi) q[1];
U1q(0.604232209210521*pi,0.9296841110039011*pi) q[2];
U1q(0.629636596900237*pi,1.9632379629012995*pi) q[3];
U1q(0.450621675698822*pi,0.9580554565539998*pi) q[4];
U1q(0.639095084473779*pi,1.8546685191797003*pi) q[5];
U1q(0.15199248217045*pi,1.5182798716339008*pi) q[6];
U1q(0.663887890099701*pi,0.5995028093327015*pi) q[7];
U1q(0.598316090522975*pi,1.1363482484651009*pi) q[8];
U1q(0.891013410126119*pi,1.4162719826933987*pi) q[9];
U1q(0.319433171621672*pi,1.1728987229553987*pi) q[10];
U1q(0.555004279643727*pi,1.676428218998101*pi) q[11];
U1q(0.380298087237258*pi,0.2821274312673996*pi) q[12];
U1q(0.699210651676678*pi,0.7178989540793985*pi) q[13];
U1q(0.346581598429433*pi,0.5964576607608016*pi) q[14];
U1q(0.671322150568716*pi,1.2185167226970997*pi) q[15];
U1q(0.411035123864427*pi,0.1590088987702991*pi) q[16];
U1q(0.452308746155715*pi,1.0219381806107997*pi) q[17];
U1q(0.566970781483092*pi,1.0226789206698008*pi) q[18];
U1q(0.832148011665358*pi,0.40456677430140076*pi) q[19];
U1q(0.24634775558733*pi,0.13860270254039975*pi) q[20];
U1q(0.720185155269309*pi,1.8786450334106988*pi) q[21];
U1q(0.852572325964707*pi,0.26519681079500046*pi) q[22];
U1q(0.508537462310473*pi,1.1643685205633005*pi) q[23];
U1q(0.481563258365736*pi,1.0172401466012992*pi) q[24];
U1q(0.146112693034236*pi,1.8822730248080006*pi) q[25];
U1q(0.54646360142367*pi,0.4440942759704001*pi) q[26];
U1q(0.677025289787372*pi,1.8153147873393003*pi) q[27];
U1q(0.517096068145965*pi,1.8989639420331983*pi) q[28];
U1q(0.561668830132091*pi,0.3422991737136982*pi) q[29];
U1q(0.429955353821597*pi,1.7980630609886994*pi) q[30];
U1q(0.685853150086107*pi,0.21189062418260107*pi) q[31];
U1q(0.528412639560572*pi,0.03856244229719863*pi) q[32];
U1q(0.832537428184122*pi,0.6575331498531014*pi) q[33];
U1q(0.215073599039262*pi,0.5315031132593013*pi) q[34];
U1q(0.348113071322637*pi,0.2846512281085012*pi) q[35];
U1q(0.782891924426691*pi,1.1554750080411011*pi) q[36];
U1q(0.246511565491484*pi,0.731481770120201*pi) q[37];
U1q(0.308529676480625*pi,0.08559461920570044*pi) q[38];
U1q(0.816252211090736*pi,1.7152638248285008*pi) q[39];
rz(2.2242186333924003*pi) q[0];
rz(3.1460838840538017*pi) q[1];
rz(0.37099981291000006*pi) q[2];
rz(2.7263868116450993*pi) q[3];
rz(1.6231901581236983*pi) q[4];
rz(3.806403257481101*pi) q[5];
rz(0.08603933687649956*pi) q[6];
rz(1.4659906669735001*pi) q[7];
rz(3.3712784634155994*pi) q[8];
rz(1.9172085414622018*pi) q[9];
rz(3.430065047046501*pi) q[10];
rz(0.09353519863369897*pi) q[11];
rz(0.5104222830605991*pi) q[12];
rz(1.3189102672561006*pi) q[13];
rz(2.6633532761272996*pi) q[14];
rz(0.6425069901663996*pi) q[15];
rz(2.5096610120452993*pi) q[16];
rz(1.4407397280629013*pi) q[17];
rz(3.413878785673301*pi) q[18];
rz(0.05532462798619875*pi) q[19];
rz(3.968167133812699*pi) q[20];
rz(3.0883195009572013*pi) q[21];
rz(0.7685200119497004*pi) q[22];
rz(1.9268450044473013*pi) q[23];
rz(3.6016040590633995*pi) q[24];
rz(2.2785051345842007*pi) q[25];
rz(3.4862010206125014*pi) q[26];
rz(2.0407544780709017*pi) q[27];
rz(3.7014291763084994*pi) q[28];
rz(1.6424710957480002*pi) q[29];
rz(1.0160573279284009*pi) q[30];
rz(3.4585625464799*pi) q[31];
rz(2.5014470072426*pi) q[32];
rz(1.0048006470982997*pi) q[33];
rz(2.3174058033674996*pi) q[34];
rz(0.4072022303366012*pi) q[35];
rz(0.42809636150730057*pi) q[36];
rz(1.3894447925284013*pi) q[37];
rz(0.8580306562852993*pi) q[38];
rz(3.0387990678742014*pi) q[39];
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
