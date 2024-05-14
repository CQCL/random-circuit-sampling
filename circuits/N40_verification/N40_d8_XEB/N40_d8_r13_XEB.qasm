OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.381212551917517*pi,0.581530047177034*pi) q[0];
U1q(0.0608940362599913*pi,1.1568312303608361*pi) q[1];
U1q(0.575828366575931*pi,1.0077916846725*pi) q[2];
U1q(0.256389458054416*pi,0.687090927671771*pi) q[3];
U1q(0.737799423649486*pi,0.483940382099216*pi) q[4];
U1q(0.537768847294414*pi,0.900997920557526*pi) q[5];
U1q(0.79161785850399*pi,1.9887289342111554*pi) q[6];
U1q(0.392828718321143*pi,1.393834607415932*pi) q[7];
U1q(0.359186690112078*pi,0.35261153525707*pi) q[8];
U1q(0.0233590217539703*pi,0.5904384098670701*pi) q[9];
U1q(0.579249869876874*pi,1.9511980796677346*pi) q[10];
U1q(0.32579819679122*pi,0.53797463019608*pi) q[11];
U1q(0.395003492672816*pi,0.57460399592232*pi) q[12];
U1q(0.717182402029256*pi,0.100362181889708*pi) q[13];
U1q(0.654488969084837*pi,0.617906608940953*pi) q[14];
U1q(0.655783711795679*pi,0.658275227956878*pi) q[15];
U1q(0.690229482762168*pi,1.9412656733425322*pi) q[16];
U1q(0.32586072757796*pi,1.194021206687677*pi) q[17];
U1q(0.244808967961462*pi,1.416610465372844*pi) q[18];
U1q(0.329846090913609*pi,1.147190601294342*pi) q[19];
U1q(0.449117517164497*pi,0.49181010797351*pi) q[20];
U1q(0.463233706917515*pi,0.718430538043716*pi) q[21];
U1q(0.582559935173709*pi,1.672078042458999*pi) q[22];
U1q(0.144485761174653*pi,0.707254017426951*pi) q[23];
U1q(0.596758753354751*pi,0.0174219315561167*pi) q[24];
U1q(0.636837065728075*pi,0.00624651805695181*pi) q[25];
U1q(0.69141850204167*pi,1.9553421540164477*pi) q[26];
U1q(0.247817805492061*pi,1.2675859612811409*pi) q[27];
U1q(0.62361843427507*pi,1.764457039458051*pi) q[28];
U1q(0.444515531337728*pi,0.178373715803184*pi) q[29];
U1q(0.316376198545867*pi,0.0442740519117653*pi) q[30];
U1q(0.879206393425561*pi,0.5338011232872*pi) q[31];
U1q(0.27822772292601*pi,1.14740420903327*pi) q[32];
U1q(0.396376118299268*pi,1.480866144912955*pi) q[33];
U1q(0.878038066826223*pi,1.33847524336664*pi) q[34];
U1q(0.581846688678163*pi,1.5312910759079221*pi) q[35];
U1q(0.499237108404924*pi,1.12505181619509*pi) q[36];
U1q(0.552218513515483*pi,1.45624290591138*pi) q[37];
U1q(0.660467021604043*pi,1.684462858254188*pi) q[38];
U1q(0.94989938677693*pi,0.0409339901720241*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[3],q[15];
RZZ(0.5*pi) q[30],q[4];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[11],q[37];
RZZ(0.5*pi) q[39],q[13];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[31],q[16];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[35],q[19];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[24],q[29];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[33],q[38];
U1q(0.879963332150936*pi,1.9290164500999598*pi) q[0];
U1q(0.635589763657489*pi,0.71710097861446*pi) q[1];
U1q(0.0249131769759369*pi,0.2730823113091301*pi) q[2];
U1q(0.594399492679318*pi,0.2001718006383999*pi) q[3];
U1q(0.907612435479315*pi,1.38415125305891*pi) q[4];
U1q(0.325665858509836*pi,0.79036657429928*pi) q[5];
U1q(0.23511101707376*pi,0.1785127527674999*pi) q[6];
U1q(0.576561619132457*pi,1.376618218259746*pi) q[7];
U1q(0.466976378581512*pi,1.7812288496406299*pi) q[8];
U1q(0.71384938685172*pi,1.6937004228598997*pi) q[9];
U1q(0.545018552740233*pi,1.9260027024127502*pi) q[10];
U1q(0.708410306266067*pi,1.3548878993788702*pi) q[11];
U1q(0.43175094181037*pi,1.97227017694654*pi) q[12];
U1q(0.817019954069725*pi,0.61235207064759*pi) q[13];
U1q(0.353923068467734*pi,1.4800739106063499*pi) q[14];
U1q(0.67797841897509*pi,1.202817604834428*pi) q[15];
U1q(0.542099703037622*pi,1.62721213756924*pi) q[16];
U1q(0.41418450069758*pi,1.64121804835206*pi) q[17];
U1q(0.69848539510503*pi,0.35915458267586*pi) q[18];
U1q(0.303194572104357*pi,0.5584972458631698*pi) q[19];
U1q(0.373575199223505*pi,1.1857944373990699*pi) q[20];
U1q(0.49381996090047*pi,0.7534746795326399*pi) q[21];
U1q(0.279116594802841*pi,0.7911676095267199*pi) q[22];
U1q(0.324010370406113*pi,1.3328735994692398*pi) q[23];
U1q(0.420664138126863*pi,0.80295847252179*pi) q[24];
U1q(0.249765305393711*pi,0.84095835389309*pi) q[25];
U1q(0.727153315726578*pi,1.3667741862965999*pi) q[26];
U1q(0.778693204392392*pi,0.1070385930984501*pi) q[27];
U1q(0.634872219741736*pi,1.1776595722307501*pi) q[28];
U1q(0.3811325067929*pi,1.03981611690529*pi) q[29];
U1q(0.454637144995905*pi,1.3664011747640101*pi) q[30];
U1q(0.376858430793592*pi,0.00683391092737407*pi) q[31];
U1q(0.718632736599171*pi,0.8688666474525899*pi) q[32];
U1q(0.393194868422781*pi,1.5988336754930401*pi) q[33];
U1q(0.766254292659275*pi,0.580120847810093*pi) q[34];
U1q(0.275261997492355*pi,0.59814860342546*pi) q[35];
U1q(0.378398028065101*pi,1.765891867720415*pi) q[36];
U1q(0.728018946944708*pi,0.35332707707046995*pi) q[37];
U1q(0.488990478750675*pi,1.29073878212406*pi) q[38];
U1q(0.598796983286399*pi,0.6160841511527799*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[4],q[22];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[12],q[37];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[32],q[18];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[35],q[29];
RZZ(0.5*pi) q[36],q[38];
U1q(0.710618957507647*pi,1.6601781634970498*pi) q[0];
U1q(0.577078995868079*pi,0.0346991116370301*pi) q[1];
U1q(0.766505116554768*pi,0.6162348599179301*pi) q[2];
U1q(0.203045506076649*pi,1.9835464465565904*pi) q[3];
U1q(0.648935784857784*pi,0.2825479251980898*pi) q[4];
U1q(0.398240586017378*pi,0.6805862295896201*pi) q[5];
U1q(0.573134708629137*pi,1.7004341393743898*pi) q[6];
U1q(0.660391808789295*pi,1.45703087768382*pi) q[7];
U1q(0.500786275367739*pi,1.96785168556353*pi) q[8];
U1q(0.800217375707984*pi,1.2201263135407796*pi) q[9];
U1q(0.264910865801979*pi,0.57442550150359*pi) q[10];
U1q(0.701926056307113*pi,1.2712726439051796*pi) q[11];
U1q(0.613149526151385*pi,1.4601489467397597*pi) q[12];
U1q(0.544441902818864*pi,0.9480045789082698*pi) q[13];
U1q(0.719672673764114*pi,1.83919340864015*pi) q[14];
U1q(0.59770487352668*pi,0.9296235762690701*pi) q[15];
U1q(0.511960711425832*pi,1.5211507186984896*pi) q[16];
U1q(0.658852803818023*pi,1.2842381369544897*pi) q[17];
U1q(0.0648161497854283*pi,0.06511733192997982*pi) q[18];
U1q(0.705895935578579*pi,1.1333366002708702*pi) q[19];
U1q(0.391474139521846*pi,0.22406973790813023*pi) q[20];
U1q(0.820195026162089*pi,1.7886268673859096*pi) q[21];
U1q(0.429793527637808*pi,0.3021595039655596*pi) q[22];
U1q(0.345612732215349*pi,1.1272394633008398*pi) q[23];
U1q(0.526023541651724*pi,1.6788176094916798*pi) q[24];
U1q(0.394950227228123*pi,1.77789725190169*pi) q[25];
U1q(0.322886948066538*pi,0.6513123359926203*pi) q[26];
U1q(0.723622170288966*pi,0.5020292247748097*pi) q[27];
U1q(0.58359043198396*pi,1.51294426411829*pi) q[28];
U1q(0.523619901945946*pi,0.0682716411694102*pi) q[29];
U1q(0.575204664923703*pi,1.6571671202374896*pi) q[30];
U1q(0.295557675253414*pi,1.9879550986363599*pi) q[31];
U1q(0.145273196006041*pi,0.5617770960305499*pi) q[32];
U1q(0.867553371096313*pi,1.99889425976881*pi) q[33];
U1q(0.593863032635218*pi,0.0852340191953798*pi) q[34];
U1q(0.786832884496393*pi,0.5341629120757299*pi) q[35];
U1q(0.596016198342399*pi,1.272157696612978*pi) q[36];
U1q(0.945692165671817*pi,1.93716211204511*pi) q[37];
U1q(0.292348615467365*pi,1.5050933020394304*pi) q[38];
U1q(0.361337951512846*pi,0.059814514088680326*pi) q[39];
RZZ(0.5*pi) q[0],q[29];
RZZ(0.5*pi) q[1],q[16];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[36],q[3];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[17],q[5];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[24],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[32],q[19];
RZZ(0.5*pi) q[38],q[21];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[25],q[39];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[28],q[37];
U1q(0.749160609071668*pi,0.3478239018006297*pi) q[0];
U1q(0.711768284319742*pi,0.6599051509453098*pi) q[1];
U1q(0.840763015259441*pi,1.2656938619009503*pi) q[2];
U1q(0.602186931380394*pi,0.32711351356342977*pi) q[3];
U1q(0.692275814893837*pi,1.8298124897091306*pi) q[4];
U1q(0.639884073012901*pi,0.83708314881161*pi) q[5];
U1q(0.641676709530962*pi,0.6483476749448096*pi) q[6];
U1q(0.413291252713083*pi,0.25733077610137034*pi) q[7];
U1q(0.378756040156626*pi,1.6794388825033*pi) q[8];
U1q(0.0754477723199588*pi,0.3582110299044494*pi) q[9];
U1q(0.203707815670176*pi,0.47243074989490985*pi) q[10];
U1q(0.51743798852553*pi,0.003981400486449793*pi) q[11];
U1q(0.683956595681586*pi,1.3778429903906098*pi) q[12];
U1q(0.568503002210396*pi,0.09245997893723956*pi) q[13];
U1q(0.545893131215188*pi,0.8414010568168404*pi) q[14];
U1q(0.163909439749884*pi,1.4541428562769596*pi) q[15];
U1q(0.508413504886688*pi,1.6403061479692704*pi) q[16];
U1q(0.19475973380301*pi,1.4689135527986004*pi) q[17];
U1q(0.127027788746449*pi,1.91774965318882*pi) q[18];
U1q(0.394907105118172*pi,1.7835195791102603*pi) q[19];
U1q(0.525056531418743*pi,1.4333222261890501*pi) q[20];
U1q(0.507035902863612*pi,1.9016028837083399*pi) q[21];
U1q(0.282443638457679*pi,1.3558752810394097*pi) q[22];
U1q(0.669558670861998*pi,1.8224393843229496*pi) q[23];
U1q(0.689369082167667*pi,1.6878228832092796*pi) q[24];
U1q(0.280742912656355*pi,1.9152759503318704*pi) q[25];
U1q(0.718567452422249*pi,0.43076438901456005*pi) q[26];
U1q(0.720823018100793*pi,0.31679866502457*pi) q[27];
U1q(0.486700555829081*pi,1.5599224860624208*pi) q[28];
U1q(0.592589237118355*pi,0.2640970142132604*pi) q[29];
U1q(0.778722954423214*pi,0.2611063247330492*pi) q[30];
U1q(0.810514415211395*pi,1.95351599432583*pi) q[31];
U1q(0.422473406140006*pi,1.6866961825014393*pi) q[32];
U1q(0.455086159071319*pi,1.2159685323416403*pi) q[33];
U1q(0.346380532932594*pi,1.8985635414173*pi) q[34];
U1q(0.619375738724178*pi,1.0099883418042097*pi) q[35];
U1q(0.529685052691673*pi,1.0009041423524598*pi) q[36];
U1q(0.460046432356325*pi,1.6202709625971599*pi) q[37];
U1q(0.202928947382444*pi,1.13126218314207*pi) q[38];
U1q(0.337418648928246*pi,1.7163847923483893*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[19],q[2];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[27];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[20],q[15];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[34];
RZZ(0.5*pi) q[35],q[24];
RZZ(0.5*pi) q[25],q[36];
RZZ(0.5*pi) q[31],q[37];
U1q(0.66563586274135*pi,0.7797656072486703*pi) q[0];
U1q(0.52447738676045*pi,0.6321327571289004*pi) q[1];
U1q(0.892224332063638*pi,0.7654109975983499*pi) q[2];
U1q(0.138391729344077*pi,0.1706126318560104*pi) q[3];
U1q(0.848307078687435*pi,1.5960717048603996*pi) q[4];
U1q(0.413045452916743*pi,1.9647054511355098*pi) q[5];
U1q(0.181598224069455*pi,1.0487220241678994*pi) q[6];
U1q(0.540669937778255*pi,1.1730637034899898*pi) q[7];
U1q(0.360848889673698*pi,1.5384652788806008*pi) q[8];
U1q(0.428428792030884*pi,1.2245416005335006*pi) q[9];
U1q(0.69629308279247*pi,1.8788456607447603*pi) q[10];
U1q(0.201215409139789*pi,0.13637093171128*pi) q[11];
U1q(0.880403144692191*pi,0.07655569382539973*pi) q[12];
U1q(0.524516659256629*pi,1.8398349661569*pi) q[13];
U1q(0.220526481214237*pi,1.9444912728218995*pi) q[14];
U1q(0.259082542950475*pi,1.3588000560246893*pi) q[15];
U1q(0.780804836357753*pi,1.6843437383739897*pi) q[16];
U1q(0.631282123905464*pi,1.8476022750494892*pi) q[17];
U1q(0.575360802896857*pi,1.3161751677307993*pi) q[18];
U1q(0.592629385850337*pi,1.2733038902456304*pi) q[19];
U1q(0.485093835861465*pi,1.2648679089911798*pi) q[20];
U1q(0.385359859419647*pi,0.4262933504603499*pi) q[21];
U1q(0.136470866960262*pi,0.7497900081968005*pi) q[22];
U1q(0.86753371916518*pi,0.9343427072865094*pi) q[23];
U1q(0.205571303324283*pi,1.2111777563575306*pi) q[24];
U1q(0.251222183104378*pi,0.21780344198666057*pi) q[25];
U1q(0.320578273618568*pi,1.4257284173102995*pi) q[26];
U1q(0.351979224802689*pi,1.3048923438814892*pi) q[27];
U1q(0.561278762109676*pi,0.5269295493925998*pi) q[28];
U1q(0.139052813169561*pi,0.7702267180431992*pi) q[29];
U1q(0.874024487851228*pi,0.10915568944080078*pi) q[30];
U1q(0.317751197099575*pi,1.3772942829350896*pi) q[31];
U1q(0.422540900057819*pi,0.7618319650221501*pi) q[32];
U1q(0.26479410781752*pi,0.2497773901841196*pi) q[33];
U1q(0.62860377005793*pi,0.51479135617998*pi) q[34];
U1q(0.765528555948186*pi,0.5338305678660697*pi) q[35];
U1q(0.939692756716367*pi,0.4614652850855201*pi) q[36];
U1q(0.238406920820369*pi,0.5589188966117096*pi) q[37];
U1q(0.151347243080222*pi,0.11975963202490014*pi) q[38];
U1q(0.296079358917393*pi,1.3179844178995008*pi) q[39];
RZZ(0.5*pi) q[0],q[37];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[32],q[3];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[39],q[8];
RZZ(0.5*pi) q[9],q[15];
RZZ(0.5*pi) q[10],q[35];
RZZ(0.5*pi) q[16],q[11];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[25],q[20];
RZZ(0.5*pi) q[36],q[22];
RZZ(0.5*pi) q[24],q[23];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[38],q[29];
U1q(0.377358411325736*pi,1.4326185480916998*pi) q[0];
U1q(0.266844627203147*pi,0.8438408938908992*pi) q[1];
U1q(0.63010781594781*pi,1.5625695630546899*pi) q[2];
U1q(0.472940001620752*pi,1.8897879898665995*pi) q[3];
U1q(0.451683512774422*pi,1.3979725666959997*pi) q[4];
U1q(0.459907980420482*pi,1.9803083514642008*pi) q[5];
U1q(0.155106751786217*pi,0.06813666397379947*pi) q[6];
U1q(0.2983814265402*pi,0.46704035195408*pi) q[7];
U1q(0.780966593991788*pi,0.8494092598868992*pi) q[8];
U1q(0.735409908236725*pi,1.9647945089462997*pi) q[9];
U1q(0.488535535314936*pi,0.9198956223844004*pi) q[10];
U1q(0.78919319914238*pi,0.9659864668301292*pi) q[11];
U1q(0.409328935620406*pi,0.28611019827100037*pi) q[12];
U1q(0.538602633609703*pi,0.3577709642742004*pi) q[13];
U1q(0.718998658628893*pi,1.9415614662829004*pi) q[14];
U1q(0.0972968830931641*pi,0.6221838749251702*pi) q[15];
U1q(0.201410291347572*pi,1.9079080505659007*pi) q[16];
U1q(0.761532017825548*pi,0.7814214696691995*pi) q[17];
U1q(0.667366028547296*pi,1.6114283636467999*pi) q[18];
U1q(0.413225219604095*pi,0.004885196562799621*pi) q[19];
U1q(0.75429864980203*pi,0.5716154388170906*pi) q[20];
U1q(0.606425922428038*pi,1.8907016907468002*pi) q[21];
U1q(0.38133543417454*pi,0.9461431935083002*pi) q[22];
U1q(0.594021590953114*pi,1.7516397438207996*pi) q[23];
U1q(0.743848501187902*pi,0.23601079893190047*pi) q[24];
U1q(0.55196178293693*pi,1.4139871543486997*pi) q[25];
U1q(0.503234449001446*pi,0.30052342754870054*pi) q[26];
U1q(0.539303197412343*pi,0.9975228253418003*pi) q[27];
U1q(0.294722842029284*pi,1.7028575058746007*pi) q[28];
U1q(0.60060341482021*pi,1.4619944009368009*pi) q[29];
U1q(0.742615330891347*pi,0.9987797449549003*pi) q[30];
U1q(0.611471220037259*pi,0.6726039895659106*pi) q[31];
U1q(0.536509530706954*pi,0.9742645957402001*pi) q[32];
U1q(0.314422072450665*pi,1.7595057821914004*pi) q[33];
U1q(0.536312335014715*pi,1.6724367060869598*pi) q[34];
U1q(0.106431214557031*pi,1.4976963030162*pi) q[35];
U1q(0.636513224614312*pi,0.6720374860227798*pi) q[36];
U1q(0.688533924690995*pi,0.7595217766128002*pi) q[37];
U1q(0.200189179781058*pi,0.8621269428931999*pi) q[38];
U1q(0.316918246403877*pi,0.4105507750835997*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[12],q[6];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[25],q[14];
RZZ(0.5*pi) q[38],q[15];
RZZ(0.5*pi) q[28],q[16];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[29];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[32],q[27];
RZZ(0.5*pi) q[39],q[35];
RZZ(0.5*pi) q[36],q[37];
U1q(0.18691791579079*pi,0.31982646150160043*pi) q[0];
U1q(0.501536178889205*pi,1.3382030894334989*pi) q[1];
U1q(0.703167661109934*pi,1.6182968765336696*pi) q[2];
U1q(0.730755711382571*pi,0.6588886334274999*pi) q[3];
U1q(0.665575133608045*pi,0.09433643379099976*pi) q[4];
U1q(0.506667386217634*pi,0.5004404632644004*pi) q[5];
U1q(0.183304102652214*pi,0.3292718068382001*pi) q[6];
U1q(0.597896104905209*pi,1.9186021082745999*pi) q[7];
U1q(0.450730902256688*pi,0.4729208063963988*pi) q[8];
U1q(0.103040221116067*pi,1.4331631244098002*pi) q[9];
U1q(0.557335590887707*pi,1.3889597629316004*pi) q[10];
U1q(0.725620565403626*pi,0.3879653859543808*pi) q[11];
U1q(0.074403763119722*pi,0.7490395071826015*pi) q[12];
U1q(0.285847612605982*pi,1.0428834310298996*pi) q[13];
U1q(0.687551826393428*pi,0.9048264162561992*pi) q[14];
U1q(0.413515497329138*pi,0.9513880276078002*pi) q[15];
U1q(0.865245376872092*pi,1.7527130502482997*pi) q[16];
U1q(0.510485375294041*pi,1.0299528071401998*pi) q[17];
U1q(0.655926104225614*pi,0.13144335060809986*pi) q[18];
U1q(0.192087874013663*pi,0.7167438072291006*pi) q[19];
U1q(0.650975257641286*pi,0.7455020219692994*pi) q[20];
U1q(0.693526851408751*pi,1.5603275919329*pi) q[21];
U1q(0.730050323320894*pi,1.5493357122226001*pi) q[22];
U1q(0.731013628901175*pi,0.07193147689399915*pi) q[23];
U1q(0.259076155625327*pi,0.9770016239941004*pi) q[24];
U1q(0.389862733308282*pi,1.1694257507779007*pi) q[25];
U1q(0.505612989129874*pi,0.8929791724820006*pi) q[26];
U1q(0.685536986830569*pi,1.2681895441298998*pi) q[27];
U1q(0.663032994492268*pi,1.5869662936577988*pi) q[28];
U1q(0.651985257593549*pi,0.5439294153638006*pi) q[29];
U1q(0.292357858228449*pi,0.21074172715620065*pi) q[30];
U1q(0.356641916664037*pi,1.8160895114308993*pi) q[31];
U1q(0.355123126898515*pi,1.8339039558912997*pi) q[32];
U1q(0.670146090111472*pi,1.6767973964135*pi) q[33];
U1q(0.485766323038881*pi,1.3811268098604899*pi) q[34];
U1q(0.347618735495846*pi,0.9235285098871007*pi) q[35];
U1q(0.631381765728221*pi,1.8132463748590908*pi) q[36];
U1q(0.633013825002801*pi,0.7320781894161001*pi) q[37];
U1q(0.553791658826931*pi,0.5821998209287997*pi) q[38];
U1q(0.473750494042079*pi,0.9349331821324007*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[1],q[22];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[31],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[28],q[24];
RZZ(0.5*pi) q[27],q[29];
RZZ(0.5*pi) q[30],q[39];
RZZ(0.5*pi) q[33],q[36];
RZZ(0.5*pi) q[35],q[34];
U1q(0.724019598787026*pi,1.3554189845282991*pi) q[0];
U1q(0.375539774125982*pi,1.5306394236685996*pi) q[1];
U1q(0.391566480420563*pi,1.3819758949233005*pi) q[2];
U1q(0.511770074407566*pi,1.5149024490865006*pi) q[3];
U1q(0.599428332713516*pi,1.6244764388191015*pi) q[4];
U1q(0.74757988748048*pi,0.37564931207650076*pi) q[5];
U1q(0.93546648818488*pi,0.9295402653321005*pi) q[6];
U1q(0.46218237140639*pi,1.8017126036872*pi) q[7];
U1q(0.471499122122131*pi,0.3190064494051015*pi) q[8];
U1q(0.273395981345378*pi,1.9255684360325986*pi) q[9];
U1q(0.791188442126813*pi,1.2181626938671002*pi) q[10];
U1q(0.276957599459833*pi,1.6335636465781*pi) q[11];
U1q(0.254131696942284*pi,1.9240603978044994*pi) q[12];
U1q(0.528689456838226*pi,0.8128010275679998*pi) q[13];
U1q(0.410532806616019*pi,1.7813734546557*pi) q[14];
U1q(0.820538856357243*pi,1.7895610270721*pi) q[15];
U1q(0.111047381541606*pi,0.8955699552846994*pi) q[16];
U1q(0.316032832247441*pi,1.0693434103315003*pi) q[17];
U1q(0.652183486599067*pi,0.5449668333066988*pi) q[18];
U1q(0.603102779372447*pi,0.7964975906362*pi) q[19];
U1q(0.361296942986188*pi,0.4737500673884991*pi) q[20];
U1q(0.539801327934253*pi,0.5825192915559008*pi) q[21];
U1q(0.465268864518372*pi,0.16167236710279909*pi) q[22];
U1q(0.896064236251925*pi,0.2715161833870994*pi) q[23];
U1q(0.403155076814176*pi,0.5827819863122983*pi) q[24];
U1q(0.443583297649782*pi,1.3708622554272*pi) q[25];
U1q(0.448590984988365*pi,0.25654999682780044*pi) q[26];
U1q(0.541740081318317*pi,1.9671154254249998*pi) q[27];
U1q(0.530466878090928*pi,1.1729882156321985*pi) q[28];
U1q(0.402220824132886*pi,1.5288133589979012*pi) q[29];
U1q(0.306715559469739*pi,0.25877253431259994*pi) q[30];
U1q(0.310830298128446*pi,0.7266290102718003*pi) q[31];
U1q(0.626631207000986*pi,1.0453009430539986*pi) q[32];
U1q(0.457379000429065*pi,1.4103514973784996*pi) q[33];
U1q(0.671392593500777*pi,1.2074820047812107*pi) q[34];
U1q(0.285251833416288*pi,0.19410969572900072*pi) q[35];
U1q(0.589176403282998*pi,0.5249484251330507*pi) q[36];
U1q(0.621290587382644*pi,1.0448154408741992*pi) q[37];
U1q(0.806168837234973*pi,1.0194968730321996*pi) q[38];
U1q(0.929106397586496*pi,1.7501127613668004*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[25],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[15];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[30],q[14];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[31],q[26];
RZZ(0.5*pi) q[35],q[27];
RZZ(0.5*pi) q[37],q[29];
U1q(0.0947874449017087*pi,0.9790694043130017*pi) q[0];
U1q(0.6075771593931*pi,0.20217669185160148*pi) q[1];
U1q(0.400351848086408*pi,1.9517785373811005*pi) q[2];
U1q(0.130295705134867*pi,1.3116834930954013*pi) q[3];
U1q(0.38694949783479*pi,0.7535236697601988*pi) q[4];
U1q(0.736719893849314*pi,0.5605104270328987*pi) q[5];
U1q(0.620661052663647*pi,0.5365072097031991*pi) q[6];
U1q(0.56249652096714*pi,0.5450454119468997*pi) q[7];
U1q(0.713942265895148*pi,1.7290510523073017*pi) q[8];
U1q(0.568345926763975*pi,0.12650778141109953*pi) q[9];
U1q(0.533496459393961*pi,1.546434814672299*pi) q[10];
U1q(0.205663138721468*pi,1.3870017430218997*pi) q[11];
U1q(0.43563161308537*pi,0.7357099640030995*pi) q[12];
U1q(0.752511537308282*pi,1.7888697229681014*pi) q[13];
U1q(0.532729548400955*pi,0.9945129593467996*pi) q[14];
U1q(0.780783250530814*pi,1.7162392904978994*pi) q[15];
U1q(0.319387349331794*pi,0.33430906494999846*pi) q[16];
U1q(0.263874785194188*pi,1.3875869665052996*pi) q[17];
U1q(0.428059011544904*pi,1.8200668364947994*pi) q[18];
U1q(0.63132602908229*pi,1.103832560764701*pi) q[19];
U1q(0.840848793781312*pi,0.2652640553276999*pi) q[20];
U1q(0.230399379111997*pi,1.0163482185005002*pi) q[21];
U1q(0.104844583342022*pi,0.12660097837240158*pi) q[22];
U1q(0.148717166071282*pi,0.5056464663563993*pi) q[23];
U1q(0.684124194007894*pi,1.9205969302506993*pi) q[24];
U1q(0.926233318849023*pi,1.2741335130781017*pi) q[25];
U1q(0.34054802110955*pi,1.6288765693687992*pi) q[26];
U1q(0.633167544019909*pi,0.7499609754071983*pi) q[27];
U1q(0.703528021208524*pi,1.9831276092569006*pi) q[28];
U1q(0.712938730126094*pi,0.10429927522540083*pi) q[29];
U1q(0.349388276348258*pi,0.09201955249400129*pi) q[30];
U1q(0.378236314004632*pi,0.2083898539731006*pi) q[31];
U1q(0.53549538284598*pi,0.7492925702775999*pi) q[32];
U1q(0.652194169181383*pi,0.6895726395847994*pi) q[33];
U1q(0.267786319260904*pi,0.7542072846121997*pi) q[34];
U1q(0.486806945905499*pi,1.8962418599491997*pi) q[35];
U1q(0.493664512946445*pi,0.4334288653948004*pi) q[36];
U1q(0.884758000125356*pi,1.6238416648069993*pi) q[37];
U1q(0.494119368377482*pi,0.8533303584627987*pi) q[38];
U1q(0.512917847917086*pi,0.12525579291279954*pi) q[39];
rz(0.6233653010862987*pi) q[0];
rz(1.7855953022417985*pi) q[1];
rz(2.3447095245044007*pi) q[2];
rz(3.1767624601309983*pi) q[3];
rz(0.18016715963640095*pi) q[4];
rz(2.7500745663857984*pi) q[5];
rz(3.6752149244993007*pi) q[6];
rz(3.8966169488322997*pi) q[7];
rz(3.1152522436427006*pi) q[8];
rz(0.7240388726178004*pi) q[9];
rz(0.45414114565189934*pi) q[10];
rz(2.1224782547631005*pi) q[11];
rz(1.2657927722727997*pi) q[12];
rz(0.5447816935963985*pi) q[13];
rz(1.3337255405308994*pi) q[14];
rz(3.6406371894656004*pi) q[15];
rz(2.966471511748601*pi) q[16];
rz(1.8721239466918007*pi) q[17];
rz(3.1748291355352016*pi) q[18];
rz(0.7197336221021011*pi) q[19];
rz(3.5694027977113*pi) q[20];
rz(3.3728348583534995*pi) q[21];
rz(3.4801235900253005*pi) q[22];
rz(2.7455377936069993*pi) q[23];
rz(3.4789027679409017*pi) q[24];
rz(0.37897731706339854*pi) q[25];
rz(3.8349359786485984*pi) q[26];
rz(2.359610317375001*pi) q[27];
rz(3.0298517592457017*pi) q[28];
rz(3.164889390470801*pi) q[29];
rz(2.034874951668801*pi) q[30];
rz(2.511543811598001*pi) q[31];
rz(0.9621433468124003*pi) q[32];
rz(3.7167198221104982*pi) q[33];
rz(3.0583953981173*pi) q[34];
rz(0.7369657454258984*pi) q[35];
rz(1.6780497490422999*pi) q[36];
rz(0.5015115760944013*pi) q[37];
rz(0.21528480139280148*pi) q[38];
rz(2.370502471693001*pi) q[39];
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
