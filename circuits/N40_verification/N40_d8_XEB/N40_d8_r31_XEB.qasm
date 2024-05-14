OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.350753658937291*pi,0.35867927365985*pi) q[0];
U1q(0.412175785270285*pi,1.786911548320519*pi) q[1];
U1q(0.916021204872322*pi,1.815192129163284*pi) q[2];
U1q(0.511445221326112*pi,1.520762151145715*pi) q[3];
U1q(0.579695059943204*pi,1.20514364751219*pi) q[4];
U1q(0.616414542653717*pi,0.420374164515361*pi) q[5];
U1q(0.364162624538599*pi,1.512205989001958*pi) q[6];
U1q(0.560904857061292*pi,0.583903565855843*pi) q[7];
U1q(0.526557548672748*pi,0.18535281807279*pi) q[8];
U1q(0.821881647473817*pi,1.845103972977646*pi) q[9];
U1q(0.419641789967564*pi,1.17907909615968*pi) q[10];
U1q(0.30692606696983*pi,0.50633575539449*pi) q[11];
U1q(0.337163859217915*pi,1.359235666991921*pi) q[12];
U1q(0.703540048982992*pi,0.0104052925846376*pi) q[13];
U1q(0.526488974053121*pi,0.947197357813381*pi) q[14];
U1q(0.602766108060705*pi,0.855714497948173*pi) q[15];
U1q(0.286882593762146*pi,0.341786051491923*pi) q[16];
U1q(0.874835419548746*pi,0.796309254644891*pi) q[17];
U1q(0.539451315095057*pi,0.865960452756112*pi) q[18];
U1q(0.850433806893538*pi,0.224516356546025*pi) q[19];
U1q(0.173841419499361*pi,0.803282511524362*pi) q[20];
U1q(0.709428727375531*pi,1.19316154164087*pi) q[21];
U1q(0.629285588881847*pi,1.446547223909176*pi) q[22];
U1q(0.723650764821611*pi,1.62059574451281*pi) q[23];
U1q(0.41102175672128*pi,0.325777274385974*pi) q[24];
U1q(0.576751086810474*pi,0.23846214444706*pi) q[25];
U1q(0.375419064315949*pi,0.93184371845856*pi) q[26];
U1q(0.777242345464987*pi,1.9201656046072877*pi) q[27];
U1q(0.217844056794361*pi,1.823531039305643*pi) q[28];
U1q(0.612569955434339*pi,1.54569999712275*pi) q[29];
U1q(0.751078189437742*pi,1.157216214962034*pi) q[30];
U1q(0.308428727759731*pi,0.146917728023961*pi) q[31];
U1q(0.23618214353426*pi,1.12292808654988*pi) q[32];
U1q(0.652127511108226*pi,1.271897900212252*pi) q[33];
U1q(0.867357992885832*pi,1.0089584976224821*pi) q[34];
U1q(0.160687767088327*pi,0.204497609732508*pi) q[35];
U1q(0.892063679746472*pi,0.240567628769335*pi) q[36];
U1q(0.658627977983105*pi,1.443871979082914*pi) q[37];
U1q(0.819488467554603*pi,1.63371485211561*pi) q[38];
U1q(0.814952348478056*pi,1.36020122353703*pi) q[39];
RZZ(0.5*pi) q[32],q[0];
RZZ(0.5*pi) q[1],q[20];
RZZ(0.5*pi) q[2],q[30];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[33],q[4];
RZZ(0.5*pi) q[5],q[36];
RZZ(0.5*pi) q[6],q[23];
RZZ(0.5*pi) q[24],q[7];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[27],q[12];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[37],q[14];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[35],q[16];
RZZ(0.5*pi) q[19],q[25];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[22],q[28];
RZZ(0.5*pi) q[39],q[34];
U1q(0.85643658815638*pi,1.6462810799562302*pi) q[0];
U1q(0.229475702520371*pi,0.3068579845391701*pi) q[1];
U1q(0.877226087801711*pi,1.7230506265489796*pi) q[2];
U1q(0.245544924415004*pi,1.1983898778751199*pi) q[3];
U1q(0.393457755681501*pi,0.8422943275085499*pi) q[4];
U1q(0.453211155417831*pi,0.5219088981116999*pi) q[5];
U1q(0.364275426900885*pi,0.21551417310437992*pi) q[6];
U1q(0.410546860804911*pi,0.207162367831917*pi) q[7];
U1q(0.701275897107829*pi,1.8495566169868098*pi) q[8];
U1q(0.661774124435328*pi,0.1426500716120902*pi) q[9];
U1q(0.193147023993324*pi,1.024768616811665*pi) q[10];
U1q(0.724210379421769*pi,1.9510548495072801*pi) q[11];
U1q(0.513534836658716*pi,0.8791819346430301*pi) q[12];
U1q(0.596063526757086*pi,0.27746039668998*pi) q[13];
U1q(0.221130725324561*pi,1.862770726208131*pi) q[14];
U1q(0.797352480817107*pi,1.6085092083429902*pi) q[15];
U1q(0.656615962889243*pi,0.0852356552209601*pi) q[16];
U1q(0.881508239983092*pi,0.0486849273984429*pi) q[17];
U1q(0.556277643333887*pi,1.261922453601049*pi) q[18];
U1q(0.585759051601465*pi,0.2757943482494001*pi) q[19];
U1q(0.679337093010031*pi,0.7041194158163*pi) q[20];
U1q(0.578505067386735*pi,0.241773302357628*pi) q[21];
U1q(0.555955699854728*pi,1.0566871116740901*pi) q[22];
U1q(0.648940674165021*pi,1.869330812690571*pi) q[23];
U1q(0.547707703264225*pi,0.17727952417344994*pi) q[24];
U1q(0.588095309135465*pi,0.34493401506172994*pi) q[25];
U1q(0.609857903933637*pi,0.8782301083096402*pi) q[26];
U1q(0.622481695407315*pi,0.9915074807743198*pi) q[27];
U1q(0.380133267649175*pi,1.8963631554527298*pi) q[28];
U1q(0.229046762400939*pi,0.2950880515374501*pi) q[29];
U1q(0.550939118464813*pi,0.36035339109169984*pi) q[30];
U1q(0.508668473437258*pi,1.0974732967242788*pi) q[31];
U1q(0.841008895916931*pi,1.96798477327967*pi) q[32];
U1q(0.148662070110307*pi,0.7201507396770301*pi) q[33];
U1q(0.186039316950794*pi,0.13101493162757016*pi) q[34];
U1q(0.740580290654267*pi,0.66032154453885*pi) q[35];
U1q(0.310017520029273*pi,0.7887608043552601*pi) q[36];
U1q(0.513556710300245*pi,0.09038288884232992*pi) q[37];
U1q(0.233949249696532*pi,0.73676404980599*pi) q[38];
U1q(0.374560390712146*pi,1.828401874946289*pi) q[39];
RZZ(0.5*pi) q[0],q[17];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[39],q[3];
RZZ(0.5*pi) q[15],q[4];
RZZ(0.5*pi) q[18],q[5];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[32],q[8];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[10],q[12];
RZZ(0.5*pi) q[22],q[11];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[16],q[30];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[24],q[36];
RZZ(0.5*pi) q[25],q[28];
RZZ(0.5*pi) q[29],q[38];
U1q(0.429377694045436*pi,0.37840550197725964*pi) q[0];
U1q(0.466158755154887*pi,1.7521177934817*pi) q[1];
U1q(0.395483767954612*pi,0.34946461716094035*pi) q[2];
U1q(0.268634395293213*pi,0.3860469887905902*pi) q[3];
U1q(0.21571119287718*pi,0.2529487058622699*pi) q[4];
U1q(0.42137125227692*pi,1.73957395234281*pi) q[5];
U1q(0.297570030976709*pi,0.008155537138229807*pi) q[6];
U1q(0.638829017790521*pi,0.15137912384092989*pi) q[7];
U1q(0.401355785291888*pi,0.46539282491044975*pi) q[8];
U1q(0.449929174368594*pi,1.6973058063080497*pi) q[9];
U1q(0.0478925465256172*pi,0.10278844032870982*pi) q[10];
U1q(0.657304018945333*pi,0.8286523232003602*pi) q[11];
U1q(0.847698705047352*pi,1.1310711732819598*pi) q[12];
U1q(0.835741994845783*pi,1.34610797294072*pi) q[13];
U1q(0.886529717811778*pi,1.0270843139966899*pi) q[14];
U1q(0.831383989592741*pi,1.2350483589439802*pi) q[15];
U1q(0.785395399065283*pi,0.11359740867098012*pi) q[16];
U1q(0.908771204796547*pi,1.735756539586409*pi) q[17];
U1q(0.661417868683148*pi,0.7218918245764399*pi) q[18];
U1q(0.425485431690012*pi,1.0159831858776398*pi) q[19];
U1q(0.2317827567971*pi,0.8758922791958401*pi) q[20];
U1q(0.728760121851234*pi,1.2464897068629028*pi) q[21];
U1q(0.648419382943036*pi,0.3599778516567298*pi) q[22];
U1q(0.228953309406829*pi,1.15036405746556*pi) q[23];
U1q(0.417670846497149*pi,0.48175797947051*pi) q[24];
U1q(0.487650257515143*pi,1.7979001972858804*pi) q[25];
U1q(0.351246578105775*pi,1.2657965391952901*pi) q[26];
U1q(0.818088958434255*pi,0.5852584864109804*pi) q[27];
U1q(0.143058193680213*pi,1.6014674273875196*pi) q[28];
U1q(0.284034237038486*pi,0.09639549331931985*pi) q[29];
U1q(0.954265423154663*pi,1.2715166009461099*pi) q[30];
U1q(0.134600167003404*pi,1.2900678615759897*pi) q[31];
U1q(0.0908349515973339*pi,1.4727538981517903*pi) q[32];
U1q(0.197771866202262*pi,1.5923255974900297*pi) q[33];
U1q(0.545248031124491*pi,1.4647816316153701*pi) q[34];
U1q(0.41762744464352*pi,0.38347448977101983*pi) q[35];
U1q(0.265705738507767*pi,0.7159521821908497*pi) q[36];
U1q(0.305795783927707*pi,0.08385391553593013*pi) q[37];
U1q(0.655099810873999*pi,1.8921285877685996*pi) q[38];
U1q(0.917825102756862*pi,0.025211099879979937*pi) q[39];
RZZ(0.5*pi) q[14],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[39],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[5],q[17];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[24],q[27];
RZZ(0.5*pi) q[34],q[25];
RZZ(0.5*pi) q[33],q[26];
RZZ(0.5*pi) q[32],q[30];
RZZ(0.5*pi) q[31],q[35];
U1q(0.195591124416135*pi,0.1683289552217202*pi) q[0];
U1q(0.118355910034443*pi,0.6799580130473402*pi) q[1];
U1q(0.518054178939364*pi,1.9300439982726*pi) q[2];
U1q(0.854651168446374*pi,0.29812723211460046*pi) q[3];
U1q(0.45763727781037*pi,1.9906373981045*pi) q[4];
U1q(0.0814998904829816*pi,1.2805879394900401*pi) q[5];
U1q(0.50734181546205*pi,1.3481768081959897*pi) q[6];
U1q(0.456243812643025*pi,1.51228400606976*pi) q[7];
U1q(0.478441414250177*pi,0.3327296323894702*pi) q[8];
U1q(0.689897097319896*pi,0.9716617547345408*pi) q[9];
U1q(0.773490662995721*pi,0.9931989609193499*pi) q[10];
U1q(0.117291090400032*pi,1.11180664434867*pi) q[11];
U1q(0.701299988593365*pi,0.5091175768145497*pi) q[12];
U1q(0.186567546403249*pi,1.0648199278835202*pi) q[13];
U1q(0.667092279428412*pi,0.6997010241426*pi) q[14];
U1q(0.846588772594664*pi,0.37007050771116035*pi) q[15];
U1q(0.523092932699265*pi,0.9934884973590004*pi) q[16];
U1q(0.151906633150016*pi,0.016183774372009907*pi) q[17];
U1q(0.220737022912272*pi,1.2916678311542*pi) q[18];
U1q(0.772888749909605*pi,0.15296217262990996*pi) q[19];
U1q(0.40483035385386*pi,1.3980035554312398*pi) q[20];
U1q(0.515157935546644*pi,1.8761112593901799*pi) q[21];
U1q(0.506413081428795*pi,1.7842801940932507*pi) q[22];
U1q(0.369217498430467*pi,1.0897760539762196*pi) q[23];
U1q(0.747009548273796*pi,0.5218012182374503*pi) q[24];
U1q(0.624661894305395*pi,1.3473419098460901*pi) q[25];
U1q(0.443153676112304*pi,0.8328995147493004*pi) q[26];
U1q(0.500474287220767*pi,1.9246444451149802*pi) q[27];
U1q(0.166881896404756*pi,1.4208749676686896*pi) q[28];
U1q(0.902340205676291*pi,0.18306675780643022*pi) q[29];
U1q(0.699503409631348*pi,0.2911372161082202*pi) q[30];
U1q(0.516553520451171*pi,0.7305082562941196*pi) q[31];
U1q(0.918551117346025*pi,1.5967310861409096*pi) q[32];
U1q(0.489875185247803*pi,1.8787211517584996*pi) q[33];
U1q(0.657602314859064*pi,1.5645503270747003*pi) q[34];
U1q(0.869772042106359*pi,0.4932737583189102*pi) q[35];
U1q(0.313154752423694*pi,0.037291524153750366*pi) q[36];
U1q(0.339221659858379*pi,0.8987701096846994*pi) q[37];
U1q(0.572305227926123*pi,1.1101463554670596*pi) q[38];
U1q(0.51865714887183*pi,1.3144511405274004*pi) q[39];
RZZ(0.5*pi) q[10],q[0];
RZZ(0.5*pi) q[1],q[28];
RZZ(0.5*pi) q[37],q[2];
RZZ(0.5*pi) q[33],q[3];
RZZ(0.5*pi) q[35],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[26];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[19],q[9];
RZZ(0.5*pi) q[11],q[21];
RZZ(0.5*pi) q[13],q[17];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[27],q[16];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[24],q[31];
RZZ(0.5*pi) q[32],q[39];
RZZ(0.5*pi) q[34],q[38];
U1q(0.694257561556537*pi,0.23205544169609915*pi) q[0];
U1q(0.677324563446052*pi,1.9537587503203007*pi) q[1];
U1q(0.606882651192114*pi,1.6763414435058994*pi) q[2];
U1q(0.184644763190536*pi,0.8426448278364997*pi) q[3];
U1q(0.551418975012651*pi,1.2536235411093202*pi) q[4];
U1q(0.355523490478172*pi,1.6932781408975295*pi) q[5];
U1q(0.440524007586434*pi,1.9653916622065104*pi) q[6];
U1q(0.706795168526924*pi,1.1368250670002702*pi) q[7];
U1q(0.931994272830244*pi,1.7529373752156001*pi) q[8];
U1q(0.529890637794786*pi,1.8959380811337994*pi) q[9];
U1q(0.163956842601803*pi,1.8004236449385296*pi) q[10];
U1q(0.667231879147623*pi,0.5304018972551798*pi) q[11];
U1q(0.442864518969642*pi,0.6659475858953208*pi) q[12];
U1q(0.385198429288166*pi,0.54064729892273*pi) q[13];
U1q(0.372629212391331*pi,0.37031741858050005*pi) q[14];
U1q(0.505550929656089*pi,1.8705530486269808*pi) q[15];
U1q(0.475626500193222*pi,0.7287059207344999*pi) q[16];
U1q(0.414763614076438*pi,1.68495050340628*pi) q[17];
U1q(0.537588500308882*pi,0.9502384053774398*pi) q[18];
U1q(0.733182234736103*pi,0.8514994526830009*pi) q[19];
U1q(0.529525176990077*pi,1.7656084949462993*pi) q[20];
U1q(0.35442474971709*pi,1.5186097270631*pi) q[21];
U1q(0.426821405730845*pi,0.6858328220339995*pi) q[22];
U1q(0.49635539014894*pi,0.6281862505061593*pi) q[23];
U1q(0.3478762238682*pi,0.9723972328579897*pi) q[24];
U1q(0.266032945873398*pi,0.15180753841490002*pi) q[25];
U1q(0.202154700586547*pi,0.5704873030170496*pi) q[26];
U1q(0.721521401680004*pi,1.2503024561592007*pi) q[27];
U1q(0.313285169111474*pi,0.8742417209123392*pi) q[28];
U1q(0.55920307815652*pi,0.09902390637967962*pi) q[29];
U1q(0.193055005855584*pi,1.2422132219468303*pi) q[30];
U1q(0.417606816405422*pi,0.8623059985120207*pi) q[31];
U1q(0.535425042587724*pi,0.6042428483743993*pi) q[32];
U1q(0.418491574706119*pi,1.7427932899083007*pi) q[33];
U1q(0.768784520625971*pi,1.8352091993087996*pi) q[34];
U1q(0.606412447982466*pi,1.6477854072471505*pi) q[35];
U1q(0.0153222206018914*pi,0.7628791822154994*pi) q[36];
U1q(0.108151219936601*pi,1.0486365965202005*pi) q[37];
U1q(0.166307917969401*pi,1.4096531754835109*pi) q[38];
U1q(0.527920840995878*pi,0.31675767760201*pi) q[39];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[3],q[38];
RZZ(0.5*pi) q[19],q[4];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[13],q[20];
RZZ(0.5*pi) q[14],q[30];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[36],q[16];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[28],q[23];
RZZ(0.5*pi) q[24],q[33];
RZZ(0.5*pi) q[26],q[35];
RZZ(0.5*pi) q[32],q[37];
U1q(0.47759773236581*pi,1.0282745412583996*pi) q[0];
U1q(0.720369851566498*pi,1.1830599435518003*pi) q[1];
U1q(0.614136876946462*pi,0.6277578179184005*pi) q[2];
U1q(0.750880738272244*pi,0.16096893670130008*pi) q[3];
U1q(0.408481295083028*pi,0.6696392822810999*pi) q[4];
U1q(0.433865525011082*pi,0.7626803467017993*pi) q[5];
U1q(0.349391726346204*pi,0.24378741070733057*pi) q[6];
U1q(0.760559912862918*pi,1.0295866567989993*pi) q[7];
U1q(0.868648641847117*pi,0.14776961075889972*pi) q[8];
U1q(0.862981417140298*pi,0.20641029714469994*pi) q[9];
U1q(0.839123287233894*pi,0.8661307780924403*pi) q[10];
U1q(0.33606774637023*pi,0.7694491605400007*pi) q[11];
U1q(0.940082206617411*pi,0.7638373508443994*pi) q[12];
U1q(0.381833230852824*pi,0.9997743235982597*pi) q[13];
U1q(0.564441078317101*pi,1.4182228033065005*pi) q[14];
U1q(0.772637713389185*pi,1.2306134854944002*pi) q[15];
U1q(0.245517796557159*pi,1.0955960135932*pi) q[16];
U1q(0.587939488929881*pi,1.4981417650005904*pi) q[17];
U1q(0.595480701181665*pi,1.6312516039939098*pi) q[18];
U1q(0.443092273389728*pi,1.9411264868763993*pi) q[19];
U1q(0.974406197397542*pi,1.9598558854478991*pi) q[20];
U1q(0.657341485136002*pi,0.8325724010111006*pi) q[21];
U1q(0.70636651313057*pi,0.19613058782960024*pi) q[22];
U1q(0.125897355432555*pi,0.7654311157699993*pi) q[23];
U1q(0.584416679506207*pi,0.3399864497561005*pi) q[24];
U1q(0.766458162189846*pi,0.27532052336655966*pi) q[25];
U1q(0.660619078849156*pi,0.2062377831672002*pi) q[26];
U1q(0.489659371022416*pi,0.45517880940420063*pi) q[27];
U1q(0.416908905230274*pi,0.15250426294580066*pi) q[28];
U1q(0.541068633566893*pi,0.6442537954073195*pi) q[29];
U1q(0.521876042098303*pi,1.7658777985091998*pi) q[30];
U1q(0.657615477522376*pi,1.0781618089286997*pi) q[31];
U1q(0.528845154565487*pi,1.3507483508965006*pi) q[32];
U1q(0.166036542147552*pi,1.5114120103740003*pi) q[33];
U1q(0.135482939003865*pi,1.007997445537999*pi) q[34];
U1q(0.720478095175821*pi,0.8090149369200006*pi) q[35];
U1q(0.15597853859414*pi,0.09916755068510064*pi) q[36];
U1q(0.316995262981445*pi,0.2597245957433998*pi) q[37];
U1q(0.653171048926554*pi,0.19797477172000022*pi) q[38];
U1q(0.810676373551961*pi,0.6461564950888707*pi) q[39];
RZZ(0.5*pi) q[0],q[16];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[23],q[3];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[7],q[39];
RZZ(0.5*pi) q[28],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[35],q[12];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[24],q[14];
RZZ(0.5*pi) q[18],q[22];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[25],q[27];
RZZ(0.5*pi) q[36],q[29];
RZZ(0.5*pi) q[31],q[30];
RZZ(0.5*pi) q[32],q[38];
RZZ(0.5*pi) q[37],q[34];
U1q(0.416002397484711*pi,0.8671933747092009*pi) q[0];
U1q(0.631432931089039*pi,0.020254097741700505*pi) q[1];
U1q(0.227839719429222*pi,0.7557547884963007*pi) q[2];
U1q(0.491630470678231*pi,1.2908875045838997*pi) q[3];
U1q(0.818108709740213*pi,1.0088647256000005*pi) q[4];
U1q(0.920601468090791*pi,0.8600807978790002*pi) q[5];
U1q(0.375940537471854*pi,0.5635693896595999*pi) q[6];
U1q(0.305263834749005*pi,1.1685094712563*pi) q[7];
U1q(0.534797234828926*pi,1.3537304355560007*pi) q[8];
U1q(0.391775600897569*pi,1.5971869325843997*pi) q[9];
U1q(0.610929146842723*pi,1.1196127169049497*pi) q[10];
U1q(0.32855392686703*pi,1.1261647627857005*pi) q[11];
U1q(0.739833233951424*pi,1.5080305317772993*pi) q[12];
U1q(0.476169528591629*pi,1.9640436831203996*pi) q[13];
U1q(0.454168874267694*pi,1.1668884124444006*pi) q[14];
U1q(0.49807482334734*pi,0.0854495097329*pi) q[15];
U1q(0.363900847881133*pi,0.26417788567269973*pi) q[16];
U1q(0.169784350135462*pi,0.45403909500361994*pi) q[17];
U1q(0.293020259206868*pi,1.5958735136636992*pi) q[18];
U1q(0.496528022431087*pi,1.6074027611100004*pi) q[19];
U1q(0.416372167493292*pi,1.1394603508678003*pi) q[20];
U1q(0.19650102950519*pi,0.0773369309935994*pi) q[21];
U1q(0.470441800452775*pi,1.7997144539489014*pi) q[22];
U1q(0.929521957478682*pi,0.4786142686152992*pi) q[23];
U1q(0.565579545470273*pi,0.6915066472046991*pi) q[24];
U1q(0.345029914244573*pi,0.5050366876447008*pi) q[25];
U1q(0.111389391888815*pi,0.5998168644697*pi) q[26];
U1q(0.0997238005122306*pi,0.4843406594485007*pi) q[27];
U1q(0.555041131543021*pi,0.8200655136995003*pi) q[28];
U1q(0.770475436914067*pi,1.6170060520198994*pi) q[29];
U1q(0.544431790577983*pi,1.3104951529114999*pi) q[30];
U1q(0.569828637647241*pi,1.6080201928213995*pi) q[31];
U1q(0.368995277014419*pi,1.2659909134358003*pi) q[32];
U1q(0.26219153242358*pi,1.0231977176726996*pi) q[33];
U1q(0.851758047535778*pi,1.0540125490226018*pi) q[34];
U1q(0.175617823310317*pi,1.1398797462016006*pi) q[35];
U1q(0.719004719732141*pi,0.4773484923805995*pi) q[36];
U1q(0.255820923113841*pi,0.0976615226053994*pi) q[37];
U1q(0.83942868024472*pi,1.4257642594367006*pi) q[38];
U1q(0.573987648114089*pi,1.1643513645044*pi) q[39];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[24],q[1];
RZZ(0.5*pi) q[2],q[4];
RZZ(0.5*pi) q[28],q[3];
RZZ(0.5*pi) q[31],q[5];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[29];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[10],q[17];
RZZ(0.5*pi) q[32],q[11];
RZZ(0.5*pi) q[12],q[38];
RZZ(0.5*pi) q[37],q[13];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[18],q[15];
RZZ(0.5*pi) q[33],q[16];
RZZ(0.5*pi) q[20],q[25];
RZZ(0.5*pi) q[22],q[30];
RZZ(0.5*pi) q[23],q[36];
RZZ(0.5*pi) q[39],q[27];
U1q(0.503250016286637*pi,0.050975177702200725*pi) q[0];
U1q(0.755145350323573*pi,0.944760375965501*pi) q[1];
U1q(0.0957494960647933*pi,0.7627521525267014*pi) q[2];
U1q(0.44866562852281*pi,1.7724114682966992*pi) q[3];
U1q(0.181533852176619*pi,0.4535290746150995*pi) q[4];
U1q(0.882412530941251*pi,0.6359973064230999*pi) q[5];
U1q(0.383675990660072*pi,1.6492884106947*pi) q[6];
U1q(0.418124089889235*pi,1.1679234283061*pi) q[7];
U1q(0.711622773115927*pi,0.06546438075930006*pi) q[8];
U1q(0.518586014779269*pi,0.9064465446291994*pi) q[9];
U1q(0.242774854946483*pi,1.8705744224896002*pi) q[10];
U1q(0.135486404297593*pi,1.3528610354712*pi) q[11];
U1q(0.344436684123191*pi,0.8933374699644006*pi) q[12];
U1q(0.871905096190729*pi,1.1170250877715002*pi) q[13];
U1q(0.814479683149867*pi,1.7414819717329983*pi) q[14];
U1q(0.621110243239268*pi,0.6544902531636012*pi) q[15];
U1q(0.708052759918696*pi,0.17033793669070008*pi) q[16];
U1q(0.857911408992902*pi,0.6505966723176595*pi) q[17];
U1q(0.803664501623327*pi,1.8105163321133002*pi) q[18];
U1q(0.0395178709323814*pi,0.11851310114250069*pi) q[19];
U1q(0.403364633901806*pi,1.5772191624768013*pi) q[20];
U1q(0.353733378562852*pi,1.6267295511086992*pi) q[21];
U1q(0.638733922712662*pi,1.1487273995841996*pi) q[22];
U1q(0.441070689855401*pi,0.41080500124819963*pi) q[23];
U1q(0.923817139913418*pi,0.2620692348617002*pi) q[24];
U1q(0.607554670983431*pi,1.0310122902549992*pi) q[25];
U1q(0.830184902832953*pi,1.2172224667128013*pi) q[26];
U1q(0.498081372192979*pi,1.5414628710909994*pi) q[27];
U1q(0.284530586735644*pi,0.8678456841647986*pi) q[28];
U1q(0.0832138591240965*pi,0.5806417006934002*pi) q[29];
U1q(0.192957169774545*pi,0.4023171600814006*pi) q[30];
U1q(0.424736846867019*pi,0.7124663872513004*pi) q[31];
U1q(0.705077212436051*pi,0.1448967457105006*pi) q[32];
U1q(0.738938247701536*pi,1.5153788569169002*pi) q[33];
U1q(0.129334163107177*pi,1.8997466580066984*pi) q[34];
U1q(0.238694991848259*pi,1.2148126009765008*pi) q[35];
U1q(0.409681355989977*pi,1.2604121252404994*pi) q[36];
U1q(0.612290272022657*pi,0.3520035358854017*pi) q[37];
U1q(0.507011521957794*pi,1.2185620654091984*pi) q[38];
U1q(0.672487235818922*pi,1.6007952627475*pi) q[39];
RZZ(0.5*pi) q[7],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[21];
RZZ(0.5*pi) q[17],q[4];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[8],q[30];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[32],q[10];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[12],q[16];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[15],q[38];
RZZ(0.5*pi) q[19],q[28];
RZZ(0.5*pi) q[33],q[20];
RZZ(0.5*pi) q[23],q[27];
RZZ(0.5*pi) q[24],q[35];
RZZ(0.5*pi) q[37],q[25];
RZZ(0.5*pi) q[26],q[29];
RZZ(0.5*pi) q[34],q[36];
U1q(0.432569861876652*pi,1.0367417363222984*pi) q[0];
U1q(0.440885885009047*pi,1.435439777016601*pi) q[1];
U1q(0.46151160202351*pi,0.5312397760971983*pi) q[2];
U1q(0.653849197429147*pi,0.5371429034367985*pi) q[3];
U1q(0.41120062567009*pi,1.7369693599687999*pi) q[4];
U1q(0.913080795233853*pi,0.40149986283759986*pi) q[5];
U1q(0.702771812929364*pi,1.2276423880968004*pi) q[6];
U1q(0.657617389953494*pi,1.1518289593990012*pi) q[7];
U1q(0.645701492682919*pi,1.5157513994599014*pi) q[8];
U1q(0.28206857394106*pi,1.3746269588321987*pi) q[9];
U1q(0.677865547244212*pi,0.21651393177180012*pi) q[10];
U1q(0.401003112000129*pi,1.7646337421210987*pi) q[11];
U1q(0.572741322330448*pi,0.0644131164032995*pi) q[12];
U1q(0.290594444244465*pi,0.8730426450430997*pi) q[13];
U1q(0.0503973092018989*pi,1.0578657552728998*pi) q[14];
U1q(0.343422085516773*pi,0.04208526919720157*pi) q[15];
U1q(0.287956386576313*pi,0.9676721387196991*pi) q[16];
U1q(0.121142258399343*pi,1.7846882147435004*pi) q[17];
U1q(0.293759281132133*pi,0.040077255331899764*pi) q[18];
U1q(0.772412542576437*pi,1.6860874644168007*pi) q[19];
U1q(0.64335932391559*pi,0.6048801854784998*pi) q[20];
U1q(0.281311426610803*pi,1.824068578464999*pi) q[21];
U1q(0.364044786995487*pi,1.1585154338702992*pi) q[22];
U1q(0.379883612610795*pi,0.5747056051854997*pi) q[23];
U1q(0.120809666992071*pi,0.1202238494177017*pi) q[24];
U1q(0.684999106319562*pi,1.1358804724240983*pi) q[25];
U1q(0.57840843797625*pi,1.7906048228709004*pi) q[26];
U1q(0.520837003379486*pi,1.6968784316634995*pi) q[27];
U1q(0.553592832046308*pi,0.051674332259601385*pi) q[28];
U1q(0.497276029279397*pi,1.2607407577764*pi) q[29];
U1q(0.230382137473654*pi,1.6601330452818992*pi) q[30];
U1q(0.791133141983317*pi,1.7408588207185005*pi) q[31];
U1q(0.148268556046434*pi,0.3434375224078998*pi) q[32];
U1q(0.491434757312449*pi,1.8694641328673*pi) q[33];
U1q(0.561702951657149*pi,0.029443286852199435*pi) q[34];
U1q(0.336078411490486*pi,0.010976962951698255*pi) q[35];
U1q(0.644315085902548*pi,0.03079254672709908*pi) q[36];
U1q(0.645265005929391*pi,1.5306590747333004*pi) q[37];
U1q(0.236769774437041*pi,0.5783819121636*pi) q[38];
U1q(0.68752093247163*pi,1.9928152572834996*pi) q[39];
rz(2.4152864725719*pi) q[0];
rz(2.5768119425551*pi) q[1];
rz(3.7491228357010016*pi) q[2];
rz(2.2193045250051*pi) q[3];
rz(0.8572318871433993*pi) q[4];
rz(2.587711955654701*pi) q[5];
rz(3.6222700007522004*pi) q[6];
rz(0.36843627445729865*pi) q[7];
rz(3.6565767744347006*pi) q[8];
rz(0.09882445646000093*pi) q[9];
rz(2.4256284332168008*pi) q[10];
rz(0.7455093172951983*pi) q[11];
rz(1.1982802658639997*pi) q[12];
rz(3.5804567682569015*pi) q[13];
rz(2.591848878634501*pi) q[14];
rz(1.0507385821986013*pi) q[15];
rz(2.9158308109287994*pi) q[16];
rz(3.8205700583517004*pi) q[17];
rz(3.5415259007508*pi) q[18];
rz(0.08440136359089934*pi) q[19];
rz(2.8996299696489984*pi) q[20];
rz(0.7248609416179015*pi) q[21];
rz(1.426009030002099*pi) q[22];
rz(2.2235879186885015*pi) q[23];
rz(3.284806781239201*pi) q[24];
rz(1.2352254317717986*pi) q[25];
rz(1.5305255390579013*pi) q[26];
rz(3.1781845926436*pi) q[27];
rz(0.18887839064349876*pi) q[28];
rz(0.18429075581590126*pi) q[29];
rz(3.5613058337357*pi) q[30];
rz(1.7592866972321985*pi) q[31];
rz(0.009397222741700517*pi) q[32];
rz(0.6019031118277987*pi) q[33];
rz(0.08564516105899855*pi) q[34];
rz(3.7573773701959006*pi) q[35];
rz(2.253760473402*pi) q[36];
rz(3.623960781535299*pi) q[37];
rz(2.893208070215799*pi) q[38];
rz(2.1357924062577*pi) q[39];
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