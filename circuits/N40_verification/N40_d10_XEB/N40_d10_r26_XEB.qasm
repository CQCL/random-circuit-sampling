OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.19515816545137*pi,1.066732957513043*pi) q[0];
U1q(0.633869153363507*pi,0.376590700210345*pi) q[1];
U1q(0.252609693100743*pi,0.692345191409772*pi) q[2];
U1q(0.626109955377899*pi,1.615212775681898*pi) q[3];
U1q(0.230147293641685*pi,1.510983676437573*pi) q[4];
U1q(0.406285439917334*pi,0.586873063145814*pi) q[5];
U1q(0.14653317998691*pi,0.0139334462700105*pi) q[6];
U1q(0.710908650428805*pi,0.795781337517009*pi) q[7];
U1q(0.100929996393371*pi,0.0824480110123182*pi) q[8];
U1q(0.597060815077858*pi,1.69770014945368*pi) q[9];
U1q(0.572284263088727*pi,1.34018115711433*pi) q[10];
U1q(0.713948986630352*pi,1.06360288986226*pi) q[11];
U1q(0.513000602536106*pi,1.845699504840749*pi) q[12];
U1q(0.680424166642904*pi,1.19673060150625*pi) q[13];
U1q(0.476137025319714*pi,1.518522124744581*pi) q[14];
U1q(0.665812970632224*pi,0.642895440369421*pi) q[15];
U1q(0.59108346550892*pi,0.592300868547359*pi) q[16];
U1q(0.847307450434247*pi,1.9409825120753297*pi) q[17];
U1q(0.238155362775551*pi,0.0508853204665318*pi) q[18];
U1q(0.45696341164469*pi,0.42570728852594*pi) q[19];
U1q(0.398661430637272*pi,1.518132412933097*pi) q[20];
U1q(0.234365832392806*pi,1.252996036602485*pi) q[21];
U1q(0.86910976650853*pi,1.04110038214166*pi) q[22];
U1q(0.394943233550667*pi,1.87326289044447*pi) q[23];
U1q(0.585557715863228*pi,1.55606611399208*pi) q[24];
U1q(0.401543892482261*pi,0.9644208238741501*pi) q[25];
U1q(0.31114126650532*pi,1.850630754531078*pi) q[26];
U1q(0.487130261758073*pi,0.0194813844755107*pi) q[27];
U1q(0.520911385809945*pi,0.954650281312322*pi) q[28];
U1q(0.278125161230918*pi,0.94470199605429*pi) q[29];
U1q(0.446410233606152*pi,0.518602436916307*pi) q[30];
U1q(0.262584189780317*pi,0.661408476988868*pi) q[31];
U1q(0.254234708013558*pi,1.4305560446325352*pi) q[32];
U1q(0.689882633097678*pi,0.80117668029246*pi) q[33];
U1q(0.639014869141319*pi,0.288050384796068*pi) q[34];
U1q(0.0364735872260781*pi,1.445086388076436*pi) q[35];
U1q(0.358208724822379*pi,0.79002516623692*pi) q[36];
U1q(0.720155949182562*pi,0.445656001636126*pi) q[37];
U1q(0.77410483957112*pi,1.3042192521683*pi) q[38];
U1q(0.35894333623238*pi,0.269784837395093*pi) q[39];
RZZ(0.5*pi) q[0],q[14];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[5],q[4];
RZZ(0.5*pi) q[18],q[7];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[12],q[11];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[36],q[20];
RZZ(0.5*pi) q[21],q[34];
RZZ(0.5*pi) q[37],q[22];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[25],q[39];
RZZ(0.5*pi) q[32],q[28];
RZZ(0.5*pi) q[30],q[29];
RZZ(0.5*pi) q[35],q[38];
U1q(0.254283646501473*pi,1.29217882237689*pi) q[0];
U1q(0.101716462564326*pi,1.2596474496113*pi) q[1];
U1q(0.704820477542371*pi,0.99095371250598*pi) q[2];
U1q(0.525144208940286*pi,0.3479315949501798*pi) q[3];
U1q(0.506451668478371*pi,0.9022100170223801*pi) q[4];
U1q(0.698832203353816*pi,1.8934876106805199*pi) q[5];
U1q(0.483840217549489*pi,1.93351088256669*pi) q[6];
U1q(0.465534679789112*pi,1.082338314414426*pi) q[7];
U1q(0.431874421773419*pi,0.3417828294062*pi) q[8];
U1q(0.523457047864148*pi,1.296968901687843*pi) q[9];
U1q(0.0371455377246177*pi,1.071700889786209*pi) q[10];
U1q(0.584591766169374*pi,0.978991827319312*pi) q[11];
U1q(0.582779517503155*pi,0.7390401816804899*pi) q[12];
U1q(0.468311697045755*pi,1.4485098900231401*pi) q[13];
U1q(0.814418879089503*pi,1.56631777866005*pi) q[14];
U1q(0.789635119911799*pi,1.238650898191937*pi) q[15];
U1q(0.389274394001347*pi,1.237208131732071*pi) q[16];
U1q(0.754657453280293*pi,0.7191226475052801*pi) q[17];
U1q(0.608460587779688*pi,1.12415260264671*pi) q[18];
U1q(0.559846433036776*pi,1.9399770889199497*pi) q[19];
U1q(0.471338463618531*pi,0.005758685460439894*pi) q[20];
U1q(0.543990938591594*pi,1.1094332693352298*pi) q[21];
U1q(0.819509025132844*pi,0.248763312551806*pi) q[22];
U1q(0.634242520428664*pi,0.80638046809495*pi) q[23];
U1q(0.188004066752576*pi,0.554687124177793*pi) q[24];
U1q(0.545623474163552*pi,0.3313018402069199*pi) q[25];
U1q(0.320971657817064*pi,0.6762199584731698*pi) q[26];
U1q(0.401959680992711*pi,1.4412715429834901*pi) q[27];
U1q(0.377757214764858*pi,0.8861821984685201*pi) q[28];
U1q(0.286197337123701*pi,0.6097670848002199*pi) q[29];
U1q(0.937389210232595*pi,1.11452582871566*pi) q[30];
U1q(0.622488832646487*pi,0.20206502093377998*pi) q[31];
U1q(0.398016404478734*pi,0.027650302613199784*pi) q[32];
U1q(0.495176739402097*pi,0.26974775755690983*pi) q[33];
U1q(0.28923376591901*pi,0.5374361793539999*pi) q[34];
U1q(0.390573138808311*pi,0.14299578822416015*pi) q[35];
U1q(0.389493693891172*pi,0.22366191417602987*pi) q[36];
U1q(0.443522445951625*pi,1.096884761427048*pi) q[37];
U1q(0.291153658480882*pi,0.7650138723073301*pi) q[38];
U1q(0.674039582851572*pi,1.16244866214618*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[23],q[8];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[25],q[12];
RZZ(0.5*pi) q[18],q[13];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[21],q[33];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[37],q[31];
RZZ(0.5*pi) q[36],q[34];
RZZ(0.5*pi) q[35],q[39];
U1q(0.353333542506987*pi,1.6173712495586*pi) q[0];
U1q(0.568693461804516*pi,0.4281042657285399*pi) q[1];
U1q(0.250493793188036*pi,1.6517143406529202*pi) q[2];
U1q(0.520773841452189*pi,1.8227081812548098*pi) q[3];
U1q(0.770176069583945*pi,0.6667639711985496*pi) q[4];
U1q(0.377741703200577*pi,0.8034589386901798*pi) q[5];
U1q(0.393289181237979*pi,1.05580563845625*pi) q[6];
U1q(0.199659349380373*pi,1.2683367859642596*pi) q[7];
U1q(0.108566031121139*pi,0.11454830269224026*pi) q[8];
U1q(0.468801689582348*pi,1.9195849085392198*pi) q[9];
U1q(0.443894736514104*pi,0.005569541115229892*pi) q[10];
U1q(0.577756118429908*pi,1.1324419175828249*pi) q[11];
U1q(0.33198495801284*pi,1.64927783697736*pi) q[12];
U1q(0.264925250310935*pi,1.2670753866737696*pi) q[13];
U1q(0.271119193180316*pi,0.5923004164790804*pi) q[14];
U1q(0.359216057746499*pi,0.4030028624330799*pi) q[15];
U1q(0.507592188981079*pi,1.80810764578215*pi) q[16];
U1q(0.914821009033315*pi,1.4060911645261296*pi) q[17];
U1q(0.500293372899399*pi,0.6136082968755003*pi) q[18];
U1q(0.0305481858211161*pi,1.7856332600884302*pi) q[19];
U1q(0.707693060216269*pi,0.06813980809363995*pi) q[20];
U1q(0.775221502250673*pi,0.9643585853611496*pi) q[21];
U1q(0.316665818784927*pi,0.03733485997730002*pi) q[22];
U1q(0.726015361504781*pi,0.4607300637929099*pi) q[23];
U1q(0.264223561608223*pi,1.70571359010982*pi) q[24];
U1q(0.202518096295862*pi,1.4339111011269203*pi) q[25];
U1q(0.70088429899821*pi,0.7707199407835503*pi) q[26];
U1q(0.608869264313882*pi,1.0788904662624903*pi) q[27];
U1q(0.606101348094785*pi,0.7068751236823401*pi) q[28];
U1q(0.640959124563425*pi,1.06203179578194*pi) q[29];
U1q(0.15706304750228*pi,0.19245627750462013*pi) q[30];
U1q(0.854570143327967*pi,1.4605019009455602*pi) q[31];
U1q(0.446118157354776*pi,0.46421951254176985*pi) q[32];
U1q(0.444949242230347*pi,1.4568286775838803*pi) q[33];
U1q(0.718806218896385*pi,1.2997880831443798*pi) q[34];
U1q(0.262212144923286*pi,1.1800298931706799*pi) q[35];
U1q(0.581629992208275*pi,1.43195870710234*pi) q[36];
U1q(0.543176037174933*pi,1.70047298152046*pi) q[37];
U1q(0.72530081103857*pi,1.2778260207431904*pi) q[38];
U1q(0.194623176188536*pi,0.17843201234988015*pi) q[39];
RZZ(0.5*pi) q[0],q[1];
RZZ(0.5*pi) q[21],q[2];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[11],q[4];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[23],q[7];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[26],q[10];
RZZ(0.5*pi) q[12],q[38];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[37],q[19];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[25],q[35];
RZZ(0.5*pi) q[36],q[29];
RZZ(0.5*pi) q[30],q[32];
U1q(0.515933465273219*pi,0.9421964079815703*pi) q[0];
U1q(0.635096778967354*pi,1.4621277353733593*pi) q[1];
U1q(0.668838440392483*pi,1.2818886779193397*pi) q[2];
U1q(0.234647659628038*pi,1.0480167549130304*pi) q[3];
U1q(0.173215221718665*pi,1.3335270551906202*pi) q[4];
U1q(0.660770152950583*pi,0.2891218468308301*pi) q[5];
U1q(0.411709802743498*pi,0.6910548633269196*pi) q[6];
U1q(0.925100255758961*pi,1.9455890946825098*pi) q[7];
U1q(0.506508346048863*pi,0.6873407592091194*pi) q[8];
U1q(0.547228450059512*pi,0.2121462759472701*pi) q[9];
U1q(0.210777303159225*pi,1.02924313553053*pi) q[10];
U1q(0.465568267619564*pi,0.24070464583568985*pi) q[11];
U1q(0.173641605149028*pi,1.2684967698166103*pi) q[12];
U1q(0.264394226674464*pi,1.28756134401781*pi) q[13];
U1q(0.462831568489638*pi,1.1731971988375998*pi) q[14];
U1q(0.695176420481622*pi,0.4876904443596599*pi) q[15];
U1q(0.279825097314488*pi,1.24868752034996*pi) q[16];
U1q(0.151552877340349*pi,1.0531123470707202*pi) q[17];
U1q(0.844217592156235*pi,0.5613986318903397*pi) q[18];
U1q(0.572563074011652*pi,0.4246315819108002*pi) q[19];
U1q(0.305688108965275*pi,0.98483407745021*pi) q[20];
U1q(0.501009528722873*pi,0.9631105095266097*pi) q[21];
U1q(0.303710384762746*pi,0.7537090242432698*pi) q[22];
U1q(0.708971707982901*pi,0.5725208037242497*pi) q[23];
U1q(0.470722826364484*pi,0.34992864675565016*pi) q[24];
U1q(0.568602638141593*pi,1.1568630272007603*pi) q[25];
U1q(0.249740891308765*pi,1.0259647743285596*pi) q[26];
U1q(0.275282602034945*pi,0.3803833546199602*pi) q[27];
U1q(0.707359226818011*pi,0.00991361472642982*pi) q[28];
U1q(0.621554411537686*pi,0.5885992265117501*pi) q[29];
U1q(0.633823802002306*pi,0.5298247001219396*pi) q[30];
U1q(0.22125528644149*pi,0.2582728940941199*pi) q[31];
U1q(0.501805268670117*pi,1.47359235912276*pi) q[32];
U1q(0.58044868208105*pi,1.0374711955392*pi) q[33];
U1q(0.220028025135278*pi,0.3698331984580099*pi) q[34];
U1q(0.697554200775479*pi,1.68987376387002*pi) q[35];
U1q(0.536247871271943*pi,0.1775446324796892*pi) q[36];
U1q(0.93991286337701*pi,0.26872878485154006*pi) q[37];
U1q(0.844672129614431*pi,0.48680787989798*pi) q[38];
U1q(0.320692427932989*pi,0.7925915740769103*pi) q[39];
RZZ(0.5*pi) q[0],q[17];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[32],q[4];
RZZ(0.5*pi) q[5],q[20];
RZZ(0.5*pi) q[23],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[30],q[9];
RZZ(0.5*pi) q[21],q[11];
RZZ(0.5*pi) q[12],q[34];
RZZ(0.5*pi) q[14],q[13];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[16],q[26];
RZZ(0.5*pi) q[22],q[28];
RZZ(0.5*pi) q[27],q[39];
RZZ(0.5*pi) q[29],q[38];
RZZ(0.5*pi) q[35],q[31];
RZZ(0.5*pi) q[33],q[36];
U1q(0.310942548111139*pi,1.7683870146077005*pi) q[0];
U1q(0.698213267024709*pi,0.5145255582354*pi) q[1];
U1q(0.553278355650801*pi,0.5042386587854502*pi) q[2];
U1q(0.331531535919946*pi,0.7530682050456008*pi) q[3];
U1q(0.262361253651331*pi,1.7066101517671992*pi) q[4];
U1q(0.688662570078246*pi,0.7442678022922102*pi) q[5];
U1q(0.56805347846914*pi,0.19990380835973998*pi) q[6];
U1q(0.588887223912924*pi,0.2545086173308704*pi) q[7];
U1q(0.530012138589395*pi,0.8305237194471999*pi) q[8];
U1q(0.882196624015221*pi,0.2381359954515503*pi) q[9];
U1q(0.554484910176048*pi,1.9098726915265*pi) q[10];
U1q(0.290710728651637*pi,1.4147372031709704*pi) q[11];
U1q(0.635765800846284*pi,0.9002716141164999*pi) q[12];
U1q(0.638861785842587*pi,0.31758437992779953*pi) q[13];
U1q(0.678718097336361*pi,1.2408833204064305*pi) q[14];
U1q(0.215059860184938*pi,0.003613956174270072*pi) q[15];
U1q(0.313260853348754*pi,1.6693948259636304*pi) q[16];
U1q(0.319774106891672*pi,1.1797751468961994*pi) q[17];
U1q(0.78285791594977*pi,0.17197522373495921*pi) q[18];
U1q(0.500100921895871*pi,0.13583860205011966*pi) q[19];
U1q(0.220076187483222*pi,1.0101171138438207*pi) q[20];
U1q(0.506686699055568*pi,1.9953616084592003*pi) q[21];
U1q(0.0941490241220046*pi,1.70008242810726*pi) q[22];
U1q(0.196658371844453*pi,1.5820968128221597*pi) q[23];
U1q(0.370038680717799*pi,1.4458880897601496*pi) q[24];
U1q(0.578314014204424*pi,0.1667418918012995*pi) q[25];
U1q(0.224377247515835*pi,0.022410187473489884*pi) q[26];
U1q(0.596284998652527*pi,0.7000238696390007*pi) q[27];
U1q(0.365457614840764*pi,0.2703497178632501*pi) q[28];
U1q(0.662367050652398*pi,0.5988436062530393*pi) q[29];
U1q(0.794801442883611*pi,0.7272083461291405*pi) q[30];
U1q(0.611001623581071*pi,1.27672812839754*pi) q[31];
U1q(0.260429529498329*pi,0.11712461803750074*pi) q[32];
U1q(0.484075678626389*pi,1.5742196786458305*pi) q[33];
U1q(0.00329678224833516*pi,1.5560215734713*pi) q[34];
U1q(0.860608199671768*pi,0.5838545520234995*pi) q[35];
U1q(0.367422393742182*pi,1.9762895189925*pi) q[36];
U1q(0.331112932935632*pi,1.0430590697753903*pi) q[37];
U1q(0.677887958563835*pi,0.5531240122230896*pi) q[38];
U1q(0.441105783967477*pi,1.6851126693427805*pi) q[39];
RZZ(0.5*pi) q[15],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[2],q[22];
RZZ(0.5*pi) q[26],q[3];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[34],q[7];
RZZ(0.5*pi) q[10],q[38];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[33],q[12];
RZZ(0.5*pi) q[16],q[13];
RZZ(0.5*pi) q[27],q[17];
RZZ(0.5*pi) q[18],q[39];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[28],q[20];
RZZ(0.5*pi) q[21],q[37];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[30],q[31];
U1q(0.319290828566259*pi,1.2951690315431996*pi) q[0];
U1q(0.504924170518334*pi,0.5368505581815999*pi) q[1];
U1q(0.797209754857499*pi,1.18258161382448*pi) q[2];
U1q(0.389456923093228*pi,1.9417941391590006*pi) q[3];
U1q(0.486962752528086*pi,1.5262992610790995*pi) q[4];
U1q(0.576228505836813*pi,0.9196520273529192*pi) q[5];
U1q(0.442124002742592*pi,1.0668629571736297*pi) q[6];
U1q(0.762166491162471*pi,0.8596551157953094*pi) q[7];
U1q(0.368397893466291*pi,0.3551327779328002*pi) q[8];
U1q(0.188778605488478*pi,1.1048241374076104*pi) q[9];
U1q(0.536974207299104*pi,1.4892131561234994*pi) q[10];
U1q(0.494549179399058*pi,0.8705366821682805*pi) q[11];
U1q(0.547268452878634*pi,1.756805142247*pi) q[12];
U1q(0.811404964524936*pi,1.2961276633220002*pi) q[13];
U1q(0.512765013091127*pi,1.4296989756999992*pi) q[14];
U1q(0.525998970873489*pi,0.24892289320562*pi) q[15];
U1q(0.496548059527903*pi,0.1777404615374394*pi) q[16];
U1q(0.67821573093093*pi,1.0021373294189004*pi) q[17];
U1q(0.422556099294833*pi,0.15009873379089989*pi) q[18];
U1q(0.353216241392478*pi,0.4971508301505292*pi) q[19];
U1q(0.386449682834593*pi,0.8671836225286*pi) q[20];
U1q(0.762627631045256*pi,0.44861162535710086*pi) q[21];
U1q(0.3579822634758*pi,0.7424607253648006*pi) q[22];
U1q(0.805385992951219*pi,0.9945382887829801*pi) q[23];
U1q(0.454991840544703*pi,1.2403634300141793*pi) q[24];
U1q(0.267505561399941*pi,0.9815247559208995*pi) q[25];
U1q(0.470889559037954*pi,0.5422399894561991*pi) q[26];
U1q(0.215185987722847*pi,1.3113567137542006*pi) q[27];
U1q(0.404261862762942*pi,0.03215406594149961*pi) q[28];
U1q(0.908978804976204*pi,1.1582803039620995*pi) q[29];
U1q(0.768839761219068*pi,0.5108770540579002*pi) q[30];
U1q(0.23794100794471*pi,1.34606514473594*pi) q[31];
U1q(0.747357956857611*pi,0.2334796204622993*pi) q[32];
U1q(0.456831733624938*pi,1.2461896490134894*pi) q[33];
U1q(0.237791554062025*pi,0.8572099816159007*pi) q[34];
U1q(0.166682312204095*pi,0.9523477838062*pi) q[35];
U1q(0.188522844022598*pi,0.9967045524742986*pi) q[36];
U1q(0.49910510441177*pi,1.7274308447699802*pi) q[37];
U1q(0.208817083694674*pi,1.6953405353376994*pi) q[38];
U1q(0.751358111804462*pi,0.6261903886077995*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[1],q[9];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[7],q[10];
RZZ(0.5*pi) q[29],q[8];
RZZ(0.5*pi) q[11],q[22];
RZZ(0.5*pi) q[30],q[12];
RZZ(0.5*pi) q[26],q[13];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[15],q[21];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[33],q[18];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[25],q[32];
RZZ(0.5*pi) q[28],q[39];
RZZ(0.5*pi) q[36],q[35];
U1q(0.886494328610927*pi,0.19587724315229949*pi) q[0];
U1q(0.413015656402574*pi,0.7302679030346013*pi) q[1];
U1q(0.193602227946695*pi,0.5885689798042595*pi) q[2];
U1q(0.533855600647401*pi,0.4164834539834992*pi) q[3];
U1q(0.536378646879589*pi,1.9201090532760006*pi) q[4];
U1q(0.0559295835459207*pi,1.8098786352109997*pi) q[5];
U1q(0.738803654791962*pi,1.1848460881621996*pi) q[6];
U1q(0.537615094162339*pi,1.3760914288455002*pi) q[7];
U1q(0.474816085184617*pi,0.17735556227150084*pi) q[8];
U1q(0.371735369794059*pi,1.1886217278933806*pi) q[9];
U1q(0.694437657782084*pi,1.2800117670286006*pi) q[10];
U1q(0.24855064308778*pi,1.2214765506194993*pi) q[11];
U1q(0.294548420209806*pi,0.2367863644082*pi) q[12];
U1q(0.491464947469506*pi,1.0601518191420993*pi) q[13];
U1q(0.124484485978182*pi,0.25168390893590065*pi) q[14];
U1q(0.219023705612597*pi,1.5220989616277993*pi) q[15];
U1q(0.330109053473699*pi,1.4879132342230008*pi) q[16];
U1q(0.258134509067311*pi,1.6362702867173997*pi) q[17];
U1q(0.735606278746148*pi,1.1863033628882*pi) q[18];
U1q(0.104840715137031*pi,1.6986521015638*pi) q[19];
U1q(0.665738937573968*pi,0.9728360988119*pi) q[20];
U1q(0.549232564221624*pi,0.7548414483849992*pi) q[21];
U1q(0.479371272277094*pi,0.2964552504206992*pi) q[22];
U1q(0.684193738559476*pi,0.11300964407488934*pi) q[23];
U1q(0.826760689743055*pi,1.5584267335324995*pi) q[24];
U1q(0.628499454733559*pi,1.8933569828273988*pi) q[25];
U1q(0.0982238707285686*pi,0.21850637966559994*pi) q[26];
U1q(0.771879610930402*pi,0.7665021730007986*pi) q[27];
U1q(0.504941377649579*pi,1.7965055224319002*pi) q[28];
U1q(0.412301905312355*pi,1.4081305843930991*pi) q[29];
U1q(0.784584487612256*pi,1.7429215474640998*pi) q[30];
U1q(0.41593350261222*pi,1.5626412075123*pi) q[31];
U1q(0.505987630865454*pi,1.5145695970024988*pi) q[32];
U1q(0.321633867685315*pi,1.2035004652894*pi) q[33];
U1q(0.235321127194416*pi,0.9517854448068004*pi) q[34];
U1q(0.73329634434616*pi,1.3973447861826997*pi) q[35];
U1q(0.270056118869076*pi,0.10401527003520172*pi) q[36];
U1q(0.638403784017868*pi,0.06143407376483978*pi) q[37];
U1q(0.444530486018591*pi,1.7072342500761*pi) q[38];
U1q(0.767184009549726*pi,0.35920342707609976*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[11],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[5],q[14];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[16],q[9];
RZZ(0.5*pi) q[36],q[13];
RZZ(0.5*pi) q[29],q[17];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[24],q[20];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[31],q[28];
RZZ(0.5*pi) q[33],q[39];
U1q(0.711983495056443*pi,1.9971001671874014*pi) q[0];
U1q(0.676599487714734*pi,1.9083349277389985*pi) q[1];
U1q(0.703507086002321*pi,1.8033842204630997*pi) q[2];
U1q(0.537769150410014*pi,1.2927026228694984*pi) q[3];
U1q(0.499996926191242*pi,0.7625271085863012*pi) q[4];
U1q(0.468101199686145*pi,0.8870965904196009*pi) q[5];
U1q(0.831996746878619*pi,1.8871978361959005*pi) q[6];
U1q(0.659291652501964*pi,1.3747317554365992*pi) q[7];
U1q(0.543771946854707*pi,1.4382759201078983*pi) q[8];
U1q(0.700123836505988*pi,0.9845307677095008*pi) q[9];
U1q(0.418734154566468*pi,0.0817290911658013*pi) q[10];
U1q(0.770478748494015*pi,1.8523638694279008*pi) q[11];
U1q(0.32522590706288*pi,0.37805812891890156*pi) q[12];
U1q(0.563236161003696*pi,1.2322659771383009*pi) q[13];
U1q(0.690123089629584*pi,0.18190381262679978*pi) q[14];
U1q(0.466794059521707*pi,0.7409206589044004*pi) q[15];
U1q(0.641869140416857*pi,1.7828554411929005*pi) q[16];
U1q(0.618156003191844*pi,0.5561846409853999*pi) q[17];
U1q(0.317010977938369*pi,0.38825703455379923*pi) q[18];
U1q(0.599450094400732*pi,0.8116949875533006*pi) q[19];
U1q(0.705188656252818*pi,1.4298047046790998*pi) q[20];
U1q(0.756565154722017*pi,0.8118537962345016*pi) q[21];
U1q(0.390251484273498*pi,1.5422516258146999*pi) q[22];
U1q(0.770523649375981*pi,0.20985377614240086*pi) q[23];
U1q(0.39282248165859*pi,0.36783981746519956*pi) q[24];
U1q(0.868679165179088*pi,0.7990381045400987*pi) q[25];
U1q(0.506997660214382*pi,1.1383140005872008*pi) q[26];
U1q(0.323765250549851*pi,0.8476385230809989*pi) q[27];
U1q(0.395960043181285*pi,1.7011928697024992*pi) q[28];
U1q(0.334036914980291*pi,1.188295383895099*pi) q[29];
U1q(0.244513072386719*pi,1.6202502972690986*pi) q[30];
U1q(0.338342552037976*pi,0.3064349936542001*pi) q[31];
U1q(0.41318190321811*pi,1.4280135708263018*pi) q[32];
U1q(0.709527218614113*pi,0.24913165116289981*pi) q[33];
U1q(0.515729854045999*pi,0.6251366632405997*pi) q[34];
U1q(0.492497245835982*pi,0.3182823071500991*pi) q[35];
U1q(0.853823522402833*pi,1.3095921615546011*pi) q[36];
U1q(0.523699019691206*pi,0.9818131110998998*pi) q[37];
U1q(0.392552073242142*pi,0.6058860436697007*pi) q[38];
U1q(0.681832430026173*pi,0.9447152109295001*pi) q[39];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[1],q[22];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[14],q[7];
RZZ(0.5*pi) q[25],q[8];
RZZ(0.5*pi) q[24],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[39];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[15],q[26];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[37],q[17];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[33],q[30];
RZZ(0.5*pi) q[36],q[32];
U1q(0.646919956309572*pi,1.3570501919435998*pi) q[0];
U1q(0.718763745726709*pi,1.0941723782324004*pi) q[1];
U1q(0.746369179700411*pi,1.4697355271919008*pi) q[2];
U1q(0.309900769756519*pi,1.3832887687938005*pi) q[3];
U1q(0.873140707229475*pi,1.9954870437020986*pi) q[4];
U1q(0.496276041934784*pi,1.5519491379289008*pi) q[5];
U1q(0.666926011240461*pi,0.2527217040323002*pi) q[6];
U1q(0.593163902285888*pi,0.9134501005807998*pi) q[7];
U1q(0.520894031376936*pi,0.09757110039700123*pi) q[8];
U1q(0.122452851948297*pi,0.9066348481738*pi) q[9];
U1q(0.411812218702878*pi,0.15943201891549919*pi) q[10];
U1q(0.170308701807292*pi,1.172704072851399*pi) q[11];
U1q(0.575663660564228*pi,1.2602379486123994*pi) q[12];
U1q(0.24385841946355*pi,1.7627505150074008*pi) q[13];
U1q(0.879627102136417*pi,0.8357885494271997*pi) q[14];
U1q(0.385437618697418*pi,1.9876844557127988*pi) q[15];
U1q(0.42938695353478*pi,0.18849998801000112*pi) q[16];
U1q(0.257503952151409*pi,0.3775970452877999*pi) q[17];
U1q(0.519807093848336*pi,1.6592593339021988*pi) q[18];
U1q(0.449957740027706*pi,1.8549767270389985*pi) q[19];
U1q(0.657392616685911*pi,0.8878596091138995*pi) q[20];
U1q(0.261760125018354*pi,1.4858518568160015*pi) q[21];
U1q(0.320014762055975*pi,1.9737235768843995*pi) q[22];
U1q(0.56501971287272*pi,0.4003451843201997*pi) q[23];
U1q(0.594218218582951*pi,1.4458779023290003*pi) q[24];
U1q(0.695124369695912*pi,0.6908054140094997*pi) q[25];
U1q(0.751119934952675*pi,0.38082042808829897*pi) q[26];
U1q(0.395125268509725*pi,0.3479126630219014*pi) q[27];
U1q(0.902539870184593*pi,1.5340907076927017*pi) q[28];
U1q(0.865586086282633*pi,0.9503001042761987*pi) q[29];
U1q(0.493843987707553*pi,1.5204994369001987*pi) q[30];
U1q(0.679761173684051*pi,0.06472123149480069*pi) q[31];
U1q(0.526486461095855*pi,1.1824371939452014*pi) q[32];
U1q(0.450161913450422*pi,1.1764366750565998*pi) q[33];
U1q(0.560675641674945*pi,0.015199798733000591*pi) q[34];
U1q(0.671416062892898*pi,0.9364747254205987*pi) q[35];
U1q(0.202459758151735*pi,1.0189161587015008*pi) q[36];
U1q(0.619906260149753*pi,0.9671448027671996*pi) q[37];
U1q(0.753656585934238*pi,0.5099225761873996*pi) q[38];
U1q(0.94703578293026*pi,0.5785464397345006*pi) q[39];
RZZ(0.5*pi) q[23],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[36],q[7];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[11],q[17];
RZZ(0.5*pi) q[35],q[13];
RZZ(0.5*pi) q[18],q[14];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[19],q[38];
RZZ(0.5*pi) q[21],q[22];
RZZ(0.5*pi) q[24],q[31];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[32],q[27];
U1q(0.784977707167672*pi,0.6248448722271007*pi) q[0];
U1q(0.205758212679444*pi,0.041751340714398566*pi) q[1];
U1q(0.800510768670075*pi,0.34822371435740074*pi) q[2];
U1q(0.170874973358408*pi,1.1083331006380988*pi) q[3];
U1q(0.830466641372645*pi,1.359897046870799*pi) q[4];
U1q(0.446395759325837*pi,1.0599702440806986*pi) q[5];
U1q(0.387740062930989*pi,0.5767438489127983*pi) q[6];
U1q(0.923399919522665*pi,1.0630476030180986*pi) q[7];
U1q(0.527246100465941*pi,0.8752420207758007*pi) q[8];
U1q(0.60786731450234*pi,0.050961926871899266*pi) q[9];
U1q(0.287421415794389*pi,0.3129379256291003*pi) q[10];
U1q(0.67355106880987*pi,0.792398780583401*pi) q[11];
U1q(0.0748818326467489*pi,1.9198566337615013*pi) q[12];
U1q(0.380092665196916*pi,0.6621655086413014*pi) q[13];
U1q(0.69924250132106*pi,1.4793878977966983*pi) q[14];
U1q(0.520263503246822*pi,1.0972182196444002*pi) q[15];
U1q(0.489850291007845*pi,0.36605473798719856*pi) q[16];
U1q(0.585467600620249*pi,1.5835631843632*pi) q[17];
U1q(0.492112894654934*pi,0.5087096779615017*pi) q[18];
U1q(0.441631175531595*pi,1.9951528810092007*pi) q[19];
U1q(0.0174695571512598*pi,0.7560986308807003*pi) q[20];
U1q(0.713065188866317*pi,1.2372373546455009*pi) q[21];
U1q(0.458944693493642*pi,0.7867745909318984*pi) q[22];
U1q(0.66431666441739*pi,1.6692944610440001*pi) q[23];
U1q(0.85969033296376*pi,0.7786138095007011*pi) q[24];
U1q(0.744932065553379*pi,0.4876636024509011*pi) q[25];
U1q(0.518894871008611*pi,0.8669597667246016*pi) q[26];
U1q(0.46744422602763*pi,0.35413762095999957*pi) q[27];
U1q(0.24638934303911*pi,1.4737367631659986*pi) q[28];
U1q(0.113650520485717*pi,1.532506366198799*pi) q[29];
U1q(0.776567071628147*pi,1.3880809187982983*pi) q[30];
U1q(0.333989647184391*pi,1.7027051583042017*pi) q[31];
U1q(0.72958005025269*pi,0.6580426870933991*pi) q[32];
U1q(0.483310901009765*pi,1.0510782086630002*pi) q[33];
U1q(0.669498850550396*pi,0.4148641624014999*pi) q[34];
U1q(0.440623471929366*pi,0.5938815421397017*pi) q[35];
U1q(0.276589352735567*pi,0.5559746033279005*pi) q[36];
U1q(0.891669266444804*pi,1.8560288795388011*pi) q[37];
U1q(0.682468412259166*pi,0.5223141416429016*pi) q[38];
U1q(0.68350402460588*pi,1.1970025832135*pi) q[39];
RZZ(0.5*pi) q[0],q[13];
RZZ(0.5*pi) q[1],q[24];
RZZ(0.5*pi) q[2],q[38];
RZZ(0.5*pi) q[36],q[3];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[11],q[8];
RZZ(0.5*pi) q[28],q[9];
RZZ(0.5*pi) q[30],q[10];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[32],q[14];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[33],q[17];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[21],q[19];
RZZ(0.5*pi) q[20],q[39];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[25],q[23];
U1q(0.829992116063804*pi,0.3491970964567983*pi) q[0];
U1q(0.380304511260478*pi,1.4121739000802016*pi) q[1];
U1q(0.340875513493379*pi,0.4184261749830007*pi) q[2];
U1q(0.513456292099506*pi,0.5552316513600992*pi) q[3];
U1q(0.384902003222086*pi,0.6926320299665996*pi) q[4];
U1q(0.761551689457185*pi,0.035475198169901745*pi) q[5];
U1q(0.676355276906618*pi,1.4106661847397994*pi) q[6];
U1q(0.524854948194453*pi,1.5699230725067004*pi) q[7];
U1q(0.284208441000897*pi,0.9997271939472014*pi) q[8];
U1q(0.684885895054316*pi,0.8852121633952006*pi) q[9];
U1q(0.112177478865351*pi,0.8181051296601005*pi) q[10];
U1q(0.068980124399582*pi,1.1258080566157993*pi) q[11];
U1q(0.276620105627146*pi,0.5421400970829993*pi) q[12];
U1q(0.445714080449601*pi,1.9311552394133003*pi) q[13];
U1q(0.668312619272252*pi,0.5324083834958984*pi) q[14];
U1q(0.386560994384338*pi,1.9938489359788*pi) q[15];
U1q(0.313412453618086*pi,0.4937782343883015*pi) q[16];
U1q(0.341024014175278*pi,1.8615359800263995*pi) q[17];
U1q(0.345101847433189*pi,0.9354314194622013*pi) q[18];
U1q(0.684996733046894*pi,1.8063182647284002*pi) q[19];
U1q(0.556429233344756*pi,1.9300905044517016*pi) q[20];
U1q(0.383654072560573*pi,1.1383220629684985*pi) q[21];
U1q(0.368658708468338*pi,0.8215604464839998*pi) q[22];
U1q(0.821853081217243*pi,1.6427738258939009*pi) q[23];
U1q(0.159675516703149*pi,0.08813653600979876*pi) q[24];
U1q(0.244944282736387*pi,0.5592520234644986*pi) q[25];
U1q(0.855961542132001*pi,1.1777102653663007*pi) q[26];
U1q(0.332501371715716*pi,0.48304351234019904*pi) q[27];
U1q(0.972508112232824*pi,1.6509956093209013*pi) q[28];
U1q(0.376139398874755*pi,1.2617947391639*pi) q[29];
U1q(0.13076861062357*pi,0.3286280592923987*pi) q[30];
U1q(0.188570870116406*pi,0.6042833133513987*pi) q[31];
U1q(0.465397969720045*pi,1.3540787459895007*pi) q[32];
U1q(0.563907432006732*pi,0.42710845708860035*pi) q[33];
U1q(0.452572589355965*pi,0.3140154401563997*pi) q[34];
U1q(0.757303445557896*pi,1.4008367289534007*pi) q[35];
U1q(0.78081224386967*pi,0.6899391878292*pi) q[36];
U1q(0.217718150642768*pi,0.031243207226399505*pi) q[37];
U1q(0.773818311245158*pi,0.08928170348049846*pi) q[38];
U1q(0.265004301450066*pi,0.4529945408844007*pi) q[39];
rz(2.4524857188173*pi) q[0];
rz(1.4908268148953994*pi) q[1];
rz(2.454085014611799*pi) q[2];
rz(2.217364850753299*pi) q[3];
rz(0.17664398500500056*pi) q[4];
rz(3.183873263513199*pi) q[5];
rz(1.4833458553803993*pi) q[6];
rz(1.3925490396255*pi) q[7];
rz(0.13571229372420035*pi) q[8];
rz(2.3469057296568003*pi) q[9];
rz(1.821417650343001*pi) q[10];
rz(0.6526145355266983*pi) q[11];
rz(2.7298883945337984*pi) q[12];
rz(2.1929421092827006*pi) q[13];
rz(1.1351049153051989*pi) q[14];
rz(2.1227968401160986*pi) q[15];
rz(3.639312447259499*pi) q[16];
rz(2.6056026771125005*pi) q[17];
rz(2.2709317207505*pi) q[18];
rz(0.863389654919299*pi) q[19];
rz(0.26656309481980145*pi) q[20];
rz(1.5131128340151996*pi) q[21];
rz(1.5486213098240995*pi) q[22];
rz(0.17584740994979953*pi) q[23];
rz(1.5878443611377016*pi) q[24];
rz(3.2942476719378*pi) q[25];
rz(2.035668950729299*pi) q[26];
rz(2.0052556474627004*pi) q[27];
rz(3.5940506254160987*pi) q[28];
rz(1.2483843986608996*pi) q[29];
rz(1.9462242595839996*pi) q[30];
rz(2.9744782480631002*pi) q[31];
rz(0.9962031257702009*pi) q[32];
rz(1.1042376145971993*pi) q[33];
rz(0.572754462064399*pi) q[34];
rz(1.696632138528301*pi) q[35];
rz(2.0312078172723*pi) q[36];
rz(3.584562826862701*pi) q[37];
rz(3.9039766679509995*pi) q[38];
rz(1.2990723004592013*pi) q[39];
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
