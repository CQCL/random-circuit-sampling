OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.538241296403523*pi,0.461851945662939*pi) q[0];
U1q(0.499600319829117*pi,1.618918603106542*pi) q[1];
U1q(0.569384067954039*pi,0.216925889397452*pi) q[2];
U1q(0.182405484561658*pi,0.9118002649432999*pi) q[3];
U1q(0.561471074344686*pi,0.676583300630441*pi) q[4];
U1q(0.593664844675254*pi,0.763005977758745*pi) q[5];
U1q(0.461583729097064*pi,1.9422526598857248*pi) q[6];
U1q(0.529357450509845*pi,1.736051650235812*pi) q[7];
U1q(0.481573824107547*pi,1.41804738249808*pi) q[8];
U1q(0.503765421570757*pi,1.895733637576817*pi) q[9];
U1q(0.575631114646803*pi,0.835967951624809*pi) q[10];
U1q(0.5608924347909*pi,1.669858071299781*pi) q[11];
U1q(0.824401545179936*pi,0.216566184797849*pi) q[12];
U1q(0.917018671709389*pi,1.807619046333452*pi) q[13];
U1q(0.517631095536353*pi,0.465579603508574*pi) q[14];
U1q(0.568670998747731*pi,0.776009267613103*pi) q[15];
U1q(0.504743303791045*pi,1.551166007179806*pi) q[16];
U1q(0.698912935155056*pi,1.9321009704210796*pi) q[17];
U1q(0.209820933226584*pi,1.894924453520757*pi) q[18];
U1q(0.446536293459008*pi,0.3740265221591601*pi) q[19];
U1q(0.172232816565787*pi,1.692857903164532*pi) q[20];
U1q(0.572849315054488*pi,0.542568782728198*pi) q[21];
U1q(0.888075494180355*pi,0.626718305280366*pi) q[22];
U1q(0.728000746430835*pi,0.158230354674904*pi) q[23];
U1q(0.701518072123932*pi,0.396377245597501*pi) q[24];
U1q(0.242071237711786*pi,0.037632267823893*pi) q[25];
U1q(0.498717703652738*pi,1.492337155667772*pi) q[26];
U1q(0.308419494306044*pi,0.0258779856953018*pi) q[27];
U1q(0.179556248691809*pi,1.9253284072310084*pi) q[28];
U1q(0.362120881012393*pi,1.9246578957914087*pi) q[29];
U1q(0.809853314701498*pi,1.14784387118869*pi) q[30];
U1q(0.183777776666146*pi,0.530001928163609*pi) q[31];
U1q(0.598865849731822*pi,1.546693725162811*pi) q[32];
U1q(0.769078487172168*pi,1.3520987262563*pi) q[33];
U1q(0.34952129019435*pi,1.224123086698449*pi) q[34];
U1q(0.705359755685216*pi,0.768110166983874*pi) q[35];
U1q(0.750050110142764*pi,0.0186961192210719*pi) q[36];
U1q(0.551412858140699*pi,0.840282961272999*pi) q[37];
U1q(0.754977193526651*pi,1.378587503582593*pi) q[38];
U1q(0.268719468013834*pi,0.7899631197571799*pi) q[39];
RZZ(0.5*pi) q[14],q[0];
RZZ(0.5*pi) q[34],q[1];
RZZ(0.5*pi) q[24],q[2];
RZZ(0.5*pi) q[3],q[27];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[9],q[35];
RZZ(0.5*pi) q[10],q[15];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[12],q[16];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[22],q[21];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[39],q[37];
U1q(0.618689662080386*pi,0.9880695726100599*pi) q[0];
U1q(0.0578675042295207*pi,0.45443007717289996*pi) q[1];
U1q(0.701478061854132*pi,0.73224382345213*pi) q[2];
U1q(0.305519089991894*pi,0.7055944213176399*pi) q[3];
U1q(0.364755685128975*pi,1.011605111528942*pi) q[4];
U1q(0.407185528922519*pi,0.0717999924090382*pi) q[5];
U1q(0.500037122162029*pi,1.3418072994457302*pi) q[6];
U1q(0.493071984379576*pi,1.61589348628688*pi) q[7];
U1q(0.172676410821652*pi,1.8905556407351898*pi) q[8];
U1q(0.706818285889023*pi,0.8522381928721801*pi) q[9];
U1q(0.366572074314408*pi,1.35488765370025*pi) q[10];
U1q(0.16090417856275*pi,0.7680944708756399*pi) q[11];
U1q(0.185341676925322*pi,1.2296524901831098*pi) q[12];
U1q(0.293538948700999*pi,1.8360390553997599*pi) q[13];
U1q(0.834179163985587*pi,1.384851401880467*pi) q[14];
U1q(0.640826375048897*pi,0.138009525321636*pi) q[15];
U1q(0.807608643604797*pi,1.73228150826073*pi) q[16];
U1q(0.670635919006866*pi,0.1840042071758501*pi) q[17];
U1q(0.275547775761893*pi,1.4937710218962201*pi) q[18];
U1q(0.278109834620379*pi,1.7518208690864698*pi) q[19];
U1q(0.119553508286425*pi,0.16543984151584*pi) q[20];
U1q(0.744529598524872*pi,0.993077266671951*pi) q[21];
U1q(0.335596154568606*pi,0.32618102079533995*pi) q[22];
U1q(0.34486224920036*pi,1.7977906219436801*pi) q[23];
U1q(0.493246799043655*pi,1.0975347236389998*pi) q[24];
U1q(0.840492147680407*pi,0.8541865138840201*pi) q[25];
U1q(0.684179322722789*pi,0.41301038920370003*pi) q[26];
U1q(0.876435422493336*pi,0.5792553519982002*pi) q[27];
U1q(0.424097345853688*pi,1.74892980979133*pi) q[28];
U1q(0.388505961985755*pi,0.9030758201476798*pi) q[29];
U1q(0.269506910790007*pi,0.431991717644114*pi) q[30];
U1q(0.271479141429814*pi,0.6159218167668099*pi) q[31];
U1q(0.413769774214755*pi,0.23050440310408016*pi) q[32];
U1q(0.550139710792065*pi,0.431158281105616*pi) q[33];
U1q(0.523718719984914*pi,1.40529151717725*pi) q[34];
U1q(0.911056476237583*pi,1.712980893959316*pi) q[35];
U1q(0.138574228901696*pi,1.9254267784505599*pi) q[36];
U1q(0.721234481679881*pi,1.87188094266864*pi) q[37];
U1q(0.531900762819837*pi,1.4980419108859104*pi) q[38];
U1q(0.479151788834281*pi,0.48967170450649*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[24],q[6];
RZZ(0.5*pi) q[7],q[13];
RZZ(0.5*pi) q[8],q[35];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[38],q[12];
RZZ(0.5*pi) q[14],q[23];
RZZ(0.5*pi) q[25],q[16];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[28],q[36];
RZZ(0.5*pi) q[29],q[37];
RZZ(0.5*pi) q[30],q[31];
U1q(0.654623577052813*pi,0.5924137368256202*pi) q[0];
U1q(0.674112833034916*pi,0.6744318044635103*pi) q[1];
U1q(0.550265799201427*pi,0.8449539142707696*pi) q[2];
U1q(0.674090967163101*pi,0.5926458723563304*pi) q[3];
U1q(0.440207322493934*pi,1.4091594268693801*pi) q[4];
U1q(0.680198155282664*pi,0.95131769351315*pi) q[5];
U1q(0.653605485634002*pi,1.45797198009741*pi) q[6];
U1q(0.184606608428958*pi,1.6526220248482604*pi) q[7];
U1q(0.376825873699185*pi,0.7102007185763801*pi) q[8];
U1q(0.425232197017319*pi,0.5541668664020198*pi) q[9];
U1q(0.628006555351667*pi,0.8847673621562602*pi) q[10];
U1q(0.370582342902327*pi,1.1720284313053302*pi) q[11];
U1q(0.776955898690114*pi,0.18503075608699016*pi) q[12];
U1q(0.79071300404489*pi,0.39548703197679025*pi) q[13];
U1q(0.337978230195682*pi,1.0182473994747099*pi) q[14];
U1q(0.710960908344428*pi,0.74813586160668*pi) q[15];
U1q(0.205598921851831*pi,0.4431457124923499*pi) q[16];
U1q(0.760393339264938*pi,0.19991354614878976*pi) q[17];
U1q(0.686617695496604*pi,1.0430222152191302*pi) q[18];
U1q(0.110649136360043*pi,0.08168303559599988*pi) q[19];
U1q(0.370650654437965*pi,0.6556608332848999*pi) q[20];
U1q(0.508227153674424*pi,0.0523122768170483*pi) q[21];
U1q(0.429689105547513*pi,0.2863873876470602*pi) q[22];
U1q(0.833958864170249*pi,0.5709925036176102*pi) q[23];
U1q(0.686741718256708*pi,0.7138837069513202*pi) q[24];
U1q(0.590269215275525*pi,1.0909828687026897*pi) q[25];
U1q(0.864503546549792*pi,1.2372176720704*pi) q[26];
U1q(0.640940957098411*pi,0.19159672984419007*pi) q[27];
U1q(0.605473539724508*pi,1.06083357312487*pi) q[28];
U1q(0.277236510696825*pi,1.9915594578707703*pi) q[29];
U1q(0.558045977734174*pi,1.2306029110485701*pi) q[30];
U1q(0.683516424447769*pi,1.4672674688931102*pi) q[31];
U1q(0.820107120358003*pi,1.38898359693491*pi) q[32];
U1q(0.262687147455209*pi,1.9253006938268453*pi) q[33];
U1q(0.569457527654051*pi,1.00189821782046*pi) q[34];
U1q(0.734794057149207*pi,0.16618551800746006*pi) q[35];
U1q(0.460878212259945*pi,1.2637694792927396*pi) q[36];
U1q(0.586553888710194*pi,0.7165470492160999*pi) q[37];
U1q(0.529209004872028*pi,1.3945494068375996*pi) q[38];
U1q(0.163911517376009*pi,0.9716583429665402*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[18],q[1];
RZZ(0.5*pi) q[2],q[28];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[34],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[10],q[9];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[35],q[25];
U1q(0.948801468581452*pi,0.8148149935835303*pi) q[0];
U1q(0.867439590305213*pi,1.0167045225596603*pi) q[1];
U1q(0.250812775193341*pi,0.10038730529605022*pi) q[2];
U1q(0.343266048222418*pi,1.0080036882765704*pi) q[3];
U1q(0.560960176491522*pi,0.19973768055533014*pi) q[4];
U1q(0.722937771512058*pi,1.0612509691635097*pi) q[5];
U1q(0.331475192525139*pi,1.22971534440153*pi) q[6];
U1q(0.793976952836629*pi,0.040264503478260494*pi) q[7];
U1q(0.452351255321635*pi,1.1618911013803297*pi) q[8];
U1q(0.904624210426837*pi,1.7057150584648504*pi) q[9];
U1q(0.876777691581394*pi,0.5270035142594902*pi) q[10];
U1q(0.716713874696152*pi,1.82779981491527*pi) q[11];
U1q(0.48938514207887*pi,0.8698762544700003*pi) q[12];
U1q(0.702173003496355*pi,1.4511149949816797*pi) q[13];
U1q(0.332365206709247*pi,0.9929442504564703*pi) q[14];
U1q(0.280042312787304*pi,1.3210205933565602*pi) q[15];
U1q(0.143764954413149*pi,0.7420233310287898*pi) q[16];
U1q(0.706635919995335*pi,0.25606437146788075*pi) q[17];
U1q(0.881998639944538*pi,0.5853005133792504*pi) q[18];
U1q(0.454481798382576*pi,1.67649978045411*pi) q[19];
U1q(0.52959059264127*pi,0.5869915376398298*pi) q[20];
U1q(0.595596201271637*pi,1.4683630338667002*pi) q[21];
U1q(0.203558516328306*pi,0.20582626901236*pi) q[22];
U1q(0.600428383486497*pi,1.6741795950636806*pi) q[23];
U1q(0.0649880077340212*pi,1.8470271930126199*pi) q[24];
U1q(0.809257030906145*pi,1.3241586528717901*pi) q[25];
U1q(0.723986704429672*pi,1.2805717058382404*pi) q[26];
U1q(0.173721225461256*pi,0.3206848372360396*pi) q[27];
U1q(0.714302407009962*pi,0.5346064118999898*pi) q[28];
U1q(0.737157153441193*pi,1.11950846254787*pi) q[29];
U1q(0.651281477323129*pi,0.4436298436842998*pi) q[30];
U1q(0.795848495720913*pi,0.7632397883471196*pi) q[31];
U1q(0.33463813749037*pi,1.9144590474712597*pi) q[32];
U1q(0.338960907384888*pi,0.56646485036707*pi) q[33];
U1q(0.572887958605009*pi,1.7909442795620993*pi) q[34];
U1q(0.16452229685076*pi,1.5116509442868296*pi) q[35];
U1q(0.514656075993955*pi,1.3643448361929007*pi) q[36];
U1q(0.522051784008576*pi,0.17722772864986958*pi) q[37];
U1q(0.349024960218784*pi,0.0771489714589002*pi) q[38];
U1q(0.145480531112974*pi,1.0683995299712796*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[15],q[3];
RZZ(0.5*pi) q[34],q[5];
RZZ(0.5*pi) q[22],q[6];
RZZ(0.5*pi) q[7],q[31];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[10],q[13];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[17],q[32];
RZZ(0.5*pi) q[20],q[28];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[29],q[23];
RZZ(0.5*pi) q[24],q[37];
RZZ(0.5*pi) q[30],q[25];
RZZ(0.5*pi) q[39],q[38];
U1q(0.296411322679796*pi,1.3957625559637004*pi) q[0];
U1q(0.53214362133244*pi,1.5642983590701007*pi) q[1];
U1q(0.518518189626608*pi,1.5013529948667994*pi) q[2];
U1q(0.18407897968502*pi,0.8231233129224993*pi) q[3];
U1q(0.637086122531832*pi,0.8187281702634195*pi) q[4];
U1q(0.76345964901237*pi,0.4086517667743301*pi) q[5];
U1q(0.713689959114606*pi,0.4799201486959106*pi) q[6];
U1q(0.635686816320123*pi,0.7786089820669009*pi) q[7];
U1q(0.848091794442534*pi,1.9933462349786009*pi) q[8];
U1q(0.920025514663453*pi,1.8366896105367996*pi) q[9];
U1q(0.848479902731532*pi,0.0218327297471399*pi) q[10];
U1q(0.138640660438815*pi,1.4113452695492992*pi) q[11];
U1q(0.350568706159989*pi,1.7281798215572*pi) q[12];
U1q(0.891914039845886*pi,0.8891840741999193*pi) q[13];
U1q(0.268369938667455*pi,0.05264014081488977*pi) q[14];
U1q(0.613571043291748*pi,0.7743636180794606*pi) q[15];
U1q(0.574356994932423*pi,1.6018228743331004*pi) q[16];
U1q(0.539519247545646*pi,0.27491708038670026*pi) q[17];
U1q(0.204450323358861*pi,1.5286231944253998*pi) q[18];
U1q(0.753782910268007*pi,0.14532027529723912*pi) q[19];
U1q(0.846775292092161*pi,0.7824135038544*pi) q[20];
U1q(0.94100109158748*pi,1.1390858874928202*pi) q[21];
U1q(0.306474575064257*pi,1.1225659112271007*pi) q[22];
U1q(0.668979974105072*pi,1.9991281032545007*pi) q[23];
U1q(0.659213242097599*pi,1.24975196355964*pi) q[24];
U1q(0.767258703281561*pi,1.7494669945520993*pi) q[25];
U1q(0.395025810933881*pi,1.7807321570014896*pi) q[26];
U1q(0.673457007135794*pi,1.9609951168595998*pi) q[27];
U1q(0.143048620778687*pi,0.0013702049088895052*pi) q[28];
U1q(0.216024331059127*pi,1.5008303974603*pi) q[29];
U1q(0.468961330395799*pi,0.8187067129467902*pi) q[30];
U1q(0.670649192225948*pi,0.3878797239432199*pi) q[31];
U1q(0.670012598662158*pi,1.6361891492435898*pi) q[32];
U1q(0.689319167735178*pi,1.8291180194995702*pi) q[33];
U1q(0.388710237721218*pi,0.22865396243899916*pi) q[34];
U1q(0.702296006255937*pi,1.5884596501246602*pi) q[35];
U1q(0.498307825439858*pi,0.07886947730960969*pi) q[36];
U1q(0.463192249955628*pi,0.03148382520149973*pi) q[37];
U1q(0.649649014238482*pi,1.9156557208590996*pi) q[38];
U1q(0.171033834209801*pi,1.38305808920048*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[15],q[5];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[12];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[30],q[11];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[22],q[20];
RZZ(0.5*pi) q[21],q[31];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[33],q[28];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[39],q[35];
U1q(0.814183867079585*pi,1.33543035529153*pi) q[0];
U1q(0.228390397085911*pi,1.8641585371528002*pi) q[1];
U1q(0.823027023891222*pi,1.1951611258834003*pi) q[2];
U1q(0.508464078447691*pi,1.9226712205860004*pi) q[3];
U1q(0.243344428408906*pi,1.2806263829496007*pi) q[4];
U1q(0.68460055122033*pi,1.5055403398627991*pi) q[5];
U1q(0.542718091109617*pi,0.06686072141319954*pi) q[6];
U1q(0.430813751609602*pi,1.9858743883810988*pi) q[7];
U1q(0.396960372399842*pi,0.3656401061747001*pi) q[8];
U1q(0.260482120863145*pi,1.94485345957151*pi) q[9];
U1q(0.674687092755567*pi,0.9079223251546704*pi) q[10];
U1q(0.559433074568858*pi,0.5791350180716996*pi) q[11];
U1q(0.552225310314105*pi,0.2730944347598001*pi) q[12];
U1q(0.560778672173063*pi,0.17048587753759925*pi) q[13];
U1q(0.227118016660256*pi,1.3378328855602994*pi) q[14];
U1q(0.572733843289507*pi,0.14536274811660022*pi) q[15];
U1q(0.392637720501313*pi,0.22886473750419967*pi) q[16];
U1q(0.679313776829157*pi,0.07400197720659918*pi) q[17];
U1q(0.100933732781498*pi,1.9897368833801004*pi) q[18];
U1q(0.739157861726572*pi,1.8089796696740006*pi) q[19];
U1q(0.288248740212093*pi,1.9278503297155005*pi) q[20];
U1q(0.574697839826531*pi,1.01123555616533*pi) q[21];
U1q(0.653372619504522*pi,0.9307205332516002*pi) q[22];
U1q(0.244275979888275*pi,1.4490637640090007*pi) q[23];
U1q(0.475505343608958*pi,1.4828203022669602*pi) q[24];
U1q(0.492034065285175*pi,0.36818040936189966*pi) q[25];
U1q(0.568944075338524*pi,1.2856057443617193*pi) q[26];
U1q(0.799396725355389*pi,0.12409024599019958*pi) q[27];
U1q(0.66595129905859*pi,1.1600341882567005*pi) q[28];
U1q(0.552565919184006*pi,1.5647378750401*pi) q[29];
U1q(0.435452316818098*pi,0.7536287153000103*pi) q[30];
U1q(0.510693073624727*pi,1.3616314447659796*pi) q[31];
U1q(0.785302753508609*pi,0.8957736329321602*pi) q[32];
U1q(0.194456908687122*pi,0.09671184560179924*pi) q[33];
U1q(0.319200478466972*pi,1.8815153564162994*pi) q[34];
U1q(0.58451536753084*pi,1.3164688236779494*pi) q[35];
U1q(0.364091481845745*pi,0.12314946327360055*pi) q[36];
U1q(0.402123881388772*pi,0.3079963439486999*pi) q[37];
U1q(0.19275448365536*pi,1.2669093874156996*pi) q[38];
U1q(0.664020881179056*pi,1.2605735844555*pi) q[39];
RZZ(0.5*pi) q[30],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[4],q[36];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[6],q[31];
RZZ(0.5*pi) q[7],q[38];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[17],q[9];
RZZ(0.5*pi) q[10],q[23];
RZZ(0.5*pi) q[11],q[19];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[24],q[16];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[20],q[35];
RZZ(0.5*pi) q[33],q[25];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[29],q[32];
U1q(0.785597546203197*pi,0.7042228109071704*pi) q[0];
U1q(0.447372109239209*pi,1.3830636274481005*pi) q[1];
U1q(0.748576619444859*pi,1.8681629441810994*pi) q[2];
U1q(0.480339830131245*pi,0.4397682812361001*pi) q[3];
U1q(0.390065939242588*pi,0.6631906594884995*pi) q[4];
U1q(0.697315213487749*pi,0.3451159255336993*pi) q[5];
U1q(0.74947876622344*pi,1.3641791310299993*pi) q[6];
U1q(0.277308939331483*pi,1.1209314288989987*pi) q[7];
U1q(0.343010386275539*pi,0.9568917228175007*pi) q[8];
U1q(0.666153163316516*pi,1.8706145944624009*pi) q[9];
U1q(0.880639374098312*pi,1.3112950409571997*pi) q[10];
U1q(0.894497565549958*pi,1.4245730042148992*pi) q[11];
U1q(0.373958248382759*pi,0.8334428552283004*pi) q[12];
U1q(0.448419016404718*pi,1.8373330589813008*pi) q[13];
U1q(0.543608079068084*pi,1.0675932639207009*pi) q[14];
U1q(0.170808655127711*pi,0.42637918362700056*pi) q[15];
U1q(0.172419097795885*pi,1.0926432503216006*pi) q[16];
U1q(0.0524434121918732*pi,1.5341417204842003*pi) q[17];
U1q(0.948315783708266*pi,1.6238497189998995*pi) q[18];
U1q(0.744579003206697*pi,1.5694041711777*pi) q[19];
U1q(0.752839981762755*pi,0.8486880306941984*pi) q[20];
U1q(0.283871597504796*pi,1.8310187273769998*pi) q[21];
U1q(0.699886813301202*pi,1.4070404863361006*pi) q[22];
U1q(0.524304399075676*pi,0.12271280690599973*pi) q[23];
U1q(0.595748531893606*pi,0.09601422040369911*pi) q[24];
U1q(0.283202546865172*pi,0.004941414046999881*pi) q[25];
U1q(0.453698752467155*pi,1.2947933710570005*pi) q[26];
U1q(0.513586150403489*pi,0.7939954715025994*pi) q[27];
U1q(0.256997524314271*pi,1.3330479937164004*pi) q[28];
U1q(0.136705483623831*pi,0.36272810672399913*pi) q[29];
U1q(0.299092585332875*pi,0.7518009340149003*pi) q[30];
U1q(0.797240600213469*pi,1.6021133620167998*pi) q[31];
U1q(0.59801795233113*pi,1.7619938451309007*pi) q[32];
U1q(0.56542908023387*pi,0.7414987797061006*pi) q[33];
U1q(0.524235282720266*pi,0.5336501279819998*pi) q[34];
U1q(0.291585065234305*pi,0.18143842557200962*pi) q[35];
U1q(0.106789147772305*pi,1.0222652226525*pi) q[36];
U1q(0.819444585770686*pi,0.16942430896100014*pi) q[37];
U1q(0.834101568333415*pi,1.2788873731801012*pi) q[38];
U1q(0.205386508115671*pi,0.8930638667763002*pi) q[39];
RZZ(0.5*pi) q[18],q[0];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[13],q[5];
RZZ(0.5*pi) q[11],q[6];
RZZ(0.5*pi) q[8],q[28];
RZZ(0.5*pi) q[17],q[10];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[14],q[16];
RZZ(0.5*pi) q[15],q[33];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[20],q[38];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[23],q[37];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[26],q[31];
RZZ(0.5*pi) q[32],q[27];
RZZ(0.5*pi) q[30],q[29];
U1q(0.336557716977511*pi,1.8615992748739991*pi) q[0];
U1q(0.827632930447139*pi,1.9156658188058984*pi) q[1];
U1q(0.723491218828554*pi,1.9429568649023992*pi) q[2];
U1q(0.277425762948449*pi,1.3930253159639996*pi) q[3];
U1q(0.181664742810947*pi,0.3585641434020985*pi) q[4];
U1q(0.623730551084235*pi,0.2925223897774991*pi) q[5];
U1q(0.220955053145292*pi,0.5929023900519006*pi) q[6];
U1q(0.379985126978774*pi,1.3757792438422989*pi) q[7];
U1q(0.785735961787321*pi,1.1859149237779008*pi) q[8];
U1q(0.582023615385245*pi,0.20990400088050087*pi) q[9];
U1q(0.752177572157899*pi,0.4771498246883006*pi) q[10];
U1q(0.257409834836152*pi,1.3571637480642984*pi) q[11];
U1q(0.590090145340348*pi,1.153858595354599*pi) q[12];
U1q(0.803739096451017*pi,0.40785774351800086*pi) q[13];
U1q(0.326966962470815*pi,1.2766738546252014*pi) q[14];
U1q(0.6994005626288*pi,1.6259382277396988*pi) q[15];
U1q(0.416222919460084*pi,1.2847574645355984*pi) q[16];
U1q(0.364282737012242*pi,1.3224756208176984*pi) q[17];
U1q(0.2144572585529*pi,0.4357014077415009*pi) q[18];
U1q(0.784354874104341*pi,1.5256173071160006*pi) q[19];
U1q(0.378894409877368*pi,1.6449828094012986*pi) q[20];
U1q(0.295331337697836*pi,0.3192112073633009*pi) q[21];
U1q(0.733567759831878*pi,1.4621367823579003*pi) q[22];
U1q(0.289931611390366*pi,1.898989248504499*pi) q[23];
U1q(0.86103547102796*pi,0.5042085694648009*pi) q[24];
U1q(0.564007100508685*pi,0.08885789749880146*pi) q[25];
U1q(0.256320362068829*pi,1.0203875866423004*pi) q[26];
U1q(0.414433738168367*pi,1.9411958310336992*pi) q[27];
U1q(0.202994683773049*pi,1.6357062839777008*pi) q[28];
U1q(0.464446535162494*pi,0.12978029622970055*pi) q[29];
U1q(0.56266103715537*pi,0.5277235378671996*pi) q[30];
U1q(0.357014295644438*pi,1.3650131800092993*pi) q[31];
U1q(0.680346861323728*pi,1.2601093857124006*pi) q[32];
U1q(0.431561048696523*pi,1.8852644852421996*pi) q[33];
U1q(0.721183282812213*pi,1.630834335024499*pi) q[34];
U1q(0.0653953787989741*pi,0.36146372409250027*pi) q[35];
U1q(0.620658995973626*pi,1.7584254843407017*pi) q[36];
U1q(0.461807588123804*pi,1.9265569568446992*pi) q[37];
U1q(0.748514131246108*pi,1.1263750112250008*pi) q[38];
U1q(0.521442926278984*pi,1.4406424579599992*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[26],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[10],q[4];
RZZ(0.5*pi) q[11],q[5];
RZZ(0.5*pi) q[8],q[6];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[12],q[19];
RZZ(0.5*pi) q[13],q[25];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[30],q[16];
RZZ(0.5*pi) q[17],q[23];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[22],q[37];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[39],q[31];
RZZ(0.5*pi) q[38],q[35];
U1q(0.414988435267868*pi,1.8818662976529996*pi) q[0];
U1q(0.493386535353328*pi,1.2201908916936013*pi) q[1];
U1q(0.610225704105727*pi,1.9987595731482983*pi) q[2];
U1q(0.788932680661467*pi,0.4917762827559997*pi) q[3];
U1q(0.565173622477151*pi,0.6247326933472017*pi) q[4];
U1q(0.589965580374171*pi,1.2559021856056987*pi) q[5];
U1q(0.606993187751416*pi,1.8929322838049991*pi) q[6];
U1q(0.468687033411137*pi,0.7521637817546996*pi) q[7];
U1q(0.855380964871678*pi,0.4743812278695003*pi) q[8];
U1q(0.200264858821346*pi,1.9443494766554004*pi) q[9];
U1q(0.742326260146888*pi,1.1748736582087993*pi) q[10];
U1q(0.405678099018641*pi,1.4390866799480015*pi) q[11];
U1q(0.244549463008952*pi,0.1984074358529*pi) q[12];
U1q(0.731745918789358*pi,0.5431348898299007*pi) q[13];
U1q(0.416962569893925*pi,1.3739714096642999*pi) q[14];
U1q(0.579446318087784*pi,1.6918122715186001*pi) q[15];
U1q(0.157682395897063*pi,1.5748725462911999*pi) q[16];
U1q(0.699607503890823*pi,1.352733335186901*pi) q[17];
U1q(0.567874774992101*pi,1.8406911771828014*pi) q[18];
U1q(0.326720935354021*pi,1.0903585773356994*pi) q[19];
U1q(0.705895872963693*pi,1.4551187745771017*pi) q[20];
U1q(0.562736329944544*pi,0.1781566087230999*pi) q[21];
U1q(0.947171280737124*pi,1.5704579221537003*pi) q[22];
U1q(0.195933444387626*pi,0.3086842545282984*pi) q[23];
U1q(0.40904141988429*pi,1.5393913047516996*pi) q[24];
U1q(0.552401083088194*pi,1.6654639395937991*pi) q[25];
U1q(0.559975197793544*pi,1.2463903395128*pi) q[26];
U1q(0.65910772086187*pi,0.4048133838055996*pi) q[27];
U1q(0.105345105223328*pi,0.7175305420917013*pi) q[28];
U1q(0.823439708371472*pi,1.3343526140826008*pi) q[29];
U1q(0.54670773099013*pi,0.5512438974147003*pi) q[30];
U1q(0.716244349913211*pi,1.8907575548929998*pi) q[31];
U1q(0.513407748239821*pi,1.4254747347327008*pi) q[32];
U1q(0.663100652253503*pi,0.2050053403423*pi) q[33];
U1q(0.333770152604174*pi,1.4294928829110987*pi) q[34];
U1q(0.426356408017922*pi,0.9610696914018995*pi) q[35];
U1q(0.580445153389355*pi,0.763126287653801*pi) q[36];
U1q(0.234606369709962*pi,0.05719214504869896*pi) q[37];
U1q(0.14830028837762*pi,1.0024229114917986*pi) q[38];
U1q(0.796398890815695*pi,0.8608192108056016*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[34],q[2];
RZZ(0.5*pi) q[3],q[37];
RZZ(0.5*pi) q[4],q[21];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[16];
RZZ(0.5*pi) q[9],q[31];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[15],q[25];
RZZ(0.5*pi) q[24],q[19];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[39],q[23];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[32],q[35];
U1q(0.653550034643298*pi,0.5632487142210998*pi) q[0];
U1q(0.713049098520247*pi,1.5343462199403994*pi) q[1];
U1q(0.470762512429655*pi,0.8239604888137002*pi) q[2];
U1q(0.327469850527418*pi,0.06822877578370168*pi) q[3];
U1q(0.322609567311956*pi,0.6321250098541995*pi) q[4];
U1q(0.655184628673092*pi,0.8423029855200994*pi) q[5];
U1q(0.658381529775945*pi,1.0944962097813011*pi) q[6];
U1q(0.397148512844408*pi,1.6004747285116991*pi) q[7];
U1q(0.162580583062077*pi,1.4309414504770004*pi) q[8];
U1q(0.507982820168049*pi,0.6480523321692004*pi) q[9];
U1q(0.276429606377557*pi,1.642199650541901*pi) q[10];
U1q(0.294907812516113*pi,1.0778227893873016*pi) q[11];
U1q(0.529231313747851*pi,0.948267852496901*pi) q[12];
U1q(0.711429933803443*pi,1.9717557495409004*pi) q[13];
U1q(0.366947978127006*pi,1.587265706822901*pi) q[14];
U1q(0.0887148174751578*pi,1.9425086722776008*pi) q[15];
U1q(0.194697765280889*pi,1.7672064633826992*pi) q[16];
U1q(0.160277750478473*pi,1.7658467610967996*pi) q[17];
U1q(0.424525690359845*pi,0.7501926938080992*pi) q[18];
U1q(0.387355050657363*pi,0.4795860220920005*pi) q[19];
U1q(0.319246384434535*pi,1.4040249893776995*pi) q[20];
U1q(0.514509720251859*pi,1.7360695405232995*pi) q[21];
U1q(0.166351714995767*pi,0.3929257943749995*pi) q[22];
U1q(0.526521676844372*pi,0.34122495899660166*pi) q[23];
U1q(0.758329003163958*pi,1.8493833373975015*pi) q[24];
U1q(0.571810381459445*pi,0.13369705605299842*pi) q[25];
U1q(0.170532205278429*pi,1.6014604586010002*pi) q[26];
U1q(0.378896515730428*pi,0.9445009201531*pi) q[27];
U1q(0.197861703513903*pi,1.8575436881805985*pi) q[28];
U1q(0.635095479126963*pi,1.8302808879897015*pi) q[29];
U1q(0.415962834012442*pi,0.07922886168899979*pi) q[30];
U1q(0.462502086652899*pi,1.8404991112868991*pi) q[31];
U1q(0.275876842310779*pi,0.8625305282807005*pi) q[32];
U1q(0.57355317656415*pi,0.6868528021715008*pi) q[33];
U1q(0.949263429953002*pi,1.2754244362500984*pi) q[34];
U1q(0.92586785137389*pi,1.4447240734564986*pi) q[35];
U1q(0.834815323967116*pi,0.12554799063790156*pi) q[36];
U1q(0.76597718270369*pi,0.30526080147319945*pi) q[37];
U1q(0.512303422213666*pi,0.9125586437892999*pi) q[38];
U1q(0.222704182892312*pi,1.3527121385667016*pi) q[39];
RZZ(0.5*pi) q[0],q[23];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[26];
RZZ(0.5*pi) q[12],q[5];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[18],q[31];
RZZ(0.5*pi) q[32],q[20];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[34],q[33];
RZZ(0.5*pi) q[37],q[36];
U1q(0.77022495489995*pi,1.6657014515052992*pi) q[0];
U1q(0.202562603977842*pi,1.5408831678875998*pi) q[1];
U1q(0.745353085824947*pi,1.0633906994271989*pi) q[2];
U1q(0.452524545461515*pi,0.565069188344701*pi) q[3];
U1q(0.277700602571952*pi,1.1693539305283984*pi) q[4];
U1q(0.308096069704078*pi,0.6571267945963015*pi) q[5];
U1q(0.670436115299867*pi,0.8330122754378984*pi) q[6];
U1q(0.669673462135289*pi,1.1249669800659987*pi) q[7];
U1q(0.60458537384356*pi,0.40570062461069867*pi) q[8];
U1q(0.14926186527497*pi,1.3499681991673*pi) q[9];
U1q(0.124775268103142*pi,0.9802299417839002*pi) q[10];
U1q(0.833757943760382*pi,0.3531549229058015*pi) q[11];
U1q(0.785843498548918*pi,1.5530953423259994*pi) q[12];
U1q(0.519517911041718*pi,1.4082899942681983*pi) q[13];
U1q(0.529431068226636*pi,1.507576131734801*pi) q[14];
U1q(0.32543576952582*pi,1.7287400883224002*pi) q[15];
U1q(0.647265332976159*pi,1.7997631143948993*pi) q[16];
U1q(0.319201876300373*pi,1.4623873756849015*pi) q[17];
U1q(0.628308079971636*pi,0.47645073003129923*pi) q[18];
U1q(0.438833415655604*pi,1.3876189098916*pi) q[19];
U1q(0.823865901091402*pi,0.7136121863970004*pi) q[20];
U1q(0.17095151378699*pi,1.1071833438005*pi) q[21];
U1q(0.968930421353105*pi,1.0218425029272993*pi) q[22];
U1q(0.686402996877189*pi,0.6933520913455986*pi) q[23];
U1q(0.117718467516084*pi,0.11415945440310082*pi) q[24];
U1q(0.666855480597135*pi,0.44956966185539926*pi) q[25];
U1q(0.640688578552367*pi,0.19641579707860046*pi) q[26];
U1q(0.342288085524501*pi,1.5270213320027004*pi) q[27];
U1q(0.318080382095025*pi,1.0158182006192007*pi) q[28];
U1q(0.590926339472084*pi,0.24221760119879931*pi) q[29];
U1q(0.30858287994318*pi,0.8752334403211997*pi) q[30];
U1q(0.581172418000299*pi,0.16485612357270085*pi) q[31];
U1q(0.556477575637852*pi,0.14893756678340075*pi) q[32];
U1q(0.276961176421685*pi,1.1525815207872014*pi) q[33];
U1q(0.261270496602417*pi,0.22156547244869884*pi) q[34];
U1q(0.310618777470781*pi,0.2505535575866986*pi) q[35];
U1q(0.406866965989399*pi,0.6056437851750012*pi) q[36];
U1q(0.782569539755361*pi,0.06260352635860045*pi) q[37];
U1q(0.63835542501967*pi,1.423255797159399*pi) q[38];
U1q(0.814336680397695*pi,0.10107099469669834*pi) q[39];
rz(3.5722730073128*pi) q[0];
rz(2.5538352746266*pi) q[1];
rz(1.3298630891434016*pi) q[2];
rz(0.9778462458289994*pi) q[3];
rz(2.7996098512243*pi) q[4];
rz(1.2168945587852988*pi) q[5];
rz(0.3624478165703984*pi) q[6];
rz(3.965347472410599*pi) q[7];
rz(0.3628328674782004*pi) q[8];
rz(2.1899199110737015*pi) q[9];
rz(3.657144962355101*pi) q[10];
rz(0.7155331370583013*pi) q[11];
rz(2.4817132674504983*pi) q[12];
rz(0.933493788553001*pi) q[13];
rz(1.9250731155314007*pi) q[14];
rz(1.2076636903295004*pi) q[15];
rz(0.902360549826799*pi) q[16];
rz(1.6846489954046007*pi) q[17];
rz(1.6336324437739016*pi) q[18];
rz(3.4778630982231995*pi) q[19];
rz(3.127248092502999*pi) q[20];
rz(0.8211847551027986*pi) q[21];
rz(1.4944137531577013*pi) q[22];
rz(2.6166562817678987*pi) q[23];
rz(1.1604714450723996*pi) q[24];
rz(2.960211800213699*pi) q[25];
rz(0.4696253321780013*pi) q[26];
rz(1.0514344139402994*pi) q[27];
rz(1.6541464235469014*pi) q[28];
rz(3.0083084607167017*pi) q[29];
rz(0.08780326776599878*pi) q[30];
rz(3.910989002778699*pi) q[31];
rz(3.1888902117593005*pi) q[32];
rz(2.7090665548905015*pi) q[33];
rz(1.2211257989246995*pi) q[34];
rz(3.849677283458*pi) q[35];
rz(0.49576109218790165*pi) q[36];
rz(0.8167055792403985*pi) q[37];
rz(3.8538663557771002*pi) q[38];
rz(1.9084733267232998*pi) q[39];
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
