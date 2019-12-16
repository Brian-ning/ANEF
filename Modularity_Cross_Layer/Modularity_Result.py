# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:44:57 2018

@author: Brian Ning
"""
import numpy as np

# ===========================================AX==================================
# a = [[0.35180244351867623, 0.44458870554843005, 0.23741875706599536, 0.22048409411551295, 0.443380679408947, 0.3660610287980938],
# [0.4266890753998553, 0.4445793004890617, 0.2216635813353102, 0.2233076546378908, 0.4437193189861089, 0.4428817560833158],
# [0.42241740837428104, 0.4407850225717435, 0.2204675570769607, 0.2200444188951959, 0.4445507598430634, 0.400044322213419],
# [0.35606227939318563, 0.405173474507752, 0.291419720936125, 0.3344424827539915, 0.43005202267562964, 0.4436876068261497],
# [0.37750610172319077, 0.43158812621193304, 0.22205112355275508, 0.2233011109396158, 0.44463567985293595, 0.4441083674052284],
# [0.3757874398537192, 0.44411797892209603, 0.2220096301773496, 0.25351196194919495, 0.4419453758492006, 0.44459806164113025],
# [0.44134146861294654, 0.4444757589171573, 0.22208640355451967, 0.24935151043431905, 0.43834792376647025, 0.4404991941165633],
# [0.4084099378144095, 0.3886144647843297, 0.42552691511203333, 0.22965801977305686, 0.4413453005134736, 0.4385548302918336]]
# =============================================================================

# ============================================CS=================================
# a = [[0.4952978056426332, 0.4529420574219765, 0.27979338460055425, 0.5013581336378778, 0.35421427119828053, 0.5163787235462658],
# [0.42947536497208055, 0.4848058748469413, 0.2992392755671431, 0.5003041739570818, 0.4764127072337644, 0.4741252907270705],
# [0.3853659955155758, 0.3100378741549078, 0.29520056315337684, 0.5003041739570818, 0.2918934296575812, 0.4157403385842975],
# [0.5041323065663303, 0.4149257334212518, 0.2896831142059182, 0.34300930047997813, 0.3833940254793425, 0.3033443865193145],
# [0.49215488817061087, 0.5013776195519424, 0.2557207567529876, 0.28719232742801737, 0.5003041739570818, 0.45744155373394196],
# [0.3228020734339534, 0.5013776195519424, 0.2523930054373351, 0.27049432272280055, 0.3102739455811528, 0.40112336579997543],
# [0.28733119710054317, 0.5180177191232802, 0.2844900331377721, 0.3126128475923556, 0.26667834302733184, 0.25092282831788476],
# [0.5037295978909854, 0.3810341767469491, 0.28771292104799656, 0.5003041739570818, 0.4277564424177298, 0.30564056664844635]]
# =============================================================================

# ===========================================EU==================================
# a = [[0.8614630513992754, 0.8303188525737769, 0.4489333110839905, 0.8017477381454794, 0.7121621157290025, 0.8468261717833196],
# [0.7033582579858938, 0.8351148192977712, 0.43211900966648237, 0.7473500003233127, 0.6519960457452422, 0.8640432674375234],
# [0.7843165765474116, 0.8570937688676754, 0.42761583498599137, 0.4530529426596589, 0.8382455412538061, 0.8339766632560347],
# [0.5298855513323277, 0.5785598981909685, 0.43496488262790983, 0.8070231378512284, 0.8646681142099384, 0.7963139984719193],
# [0.7752043319564048, 0.4712791248704876, 0.45640797179444714, 0.7762275705076991, 0.861316953487173, 0.8194888987426251],
# [0.828748055070048, 0.7486747190442965, 0.5111418111001554, 0.8638174827235713, 0.7057995921509905, 0.7882172924615536],
# [0.830854596782974, 0.7085320551824995, 0.42740011129916916, 0.6862786883320601, 0.8628092803179581, 0.8060924620642094],
# [0.8500984640899312, 0.8499792308725904, 0.5188493533254213, 0.7121645037654536, 0.7280314513884879, 0.8565451259950372]] 
# =============================================================================

# ===========================================FAO==================================
# a = [[0.781159939914835, 0.9258642680159277, 0.5137793125544583, 0.6331929038425507, 0.887118164357323, 0.975274997434596],
# [0.8834300977590465, 0.8384947430799907, 0.4985753776060146, 0.9036517425803317, 0.9712887067776508, 0.9501130010533293],
# [0.6709513326222699, 0.9067140995060976, 0.5418205826199471, 0.9189051281749352, 0.8783189949869713, 0.8948475594869982],
# [0.6709513326222699, 0.9067140995060976, 0.5418205826199471, 0.9189051281749352, 0.8783189949869713, 0.8948475594869982],
# [0.6875237609197625, 0.9654171531844382, 0.4961936497482074, 0.8779075267323111, 0.5082853425560064, 0.7562550530941009],
# [0.6296952827888387, 0.7581061352488565, 0.5206553863315216, 0.8779075267323111, 0.9733771679617597, 0.8596343008924991],
# [0.5914873893097494, 0.8904479229940958, 0.49197028317745456, 0.9167339417995578, 0.9288104194002265, 0.9664549197805842],
# [0.5182292620432548, 0.8811012270256107, 0.4953860926859868, 0.8779075267323111, 0.9610640122896673, 0.9041025065153607],
# [0.9740102092507402, 0.8884098215094622, 0.4959345184329131, 0.9189051281749352, 0.9562935219813279, 0.9593333550144466]]
# =============================================================================


# =========================================London====================================
# a = [[0.13329563791980595, 0.12106846102870067, 0.033268649109226785, 0.06651781831609131, 0.16706409539432432, 0.12113495171667983],
# [0.12513381250955907, 0.11928429423459352, 0.06383794318079673, 0.053012277839324204, 0.1224448182698689, 0.1322304050273593],
# [0.12784308774423697, 0.13819479998492976, 0.027784215447132946, 0.07559283591409369, 0.12441047927422602, 0.12324769332721722],
# [0.12324769332721724, 0.1427838590944303, 0.06339555980526326, 0.09626425674506037, 0.0717733063813738, 0.025698403900381826],
# [0.10435411065861044, 0.13549728746031378, 0.05346137036560264, 0.03668659090840546, 0.008273143906437502, 0.029991566096831637],
# [0.07376402155761659, 0.11256848774513961, 0.05424557996475914, 0.04863616917719803, 0.005763000271330534, 0.025793073021489293],
# [0.10562961033813328, 0.11928429423459351, 0.043664835342220355, 0.04830981071703353, 0.028100940768467215, 0.07228651302083985],
# [0.11199057952455947, 0.1146577439901841, 0.0833630538697677, 0.05413697850772655, 0.06591387458515811, 0.13317403205914038]]
# =============================================================================

# =========================================PFF====================================
# a = [[0.09239766081871346, 0.39298245614035093, 0.19122807017543864, 0.24736842105263157, 0.14035087719298248, 0.08713450292397665],
# [0.23625730994152047, 0.18187134502923977, 0.25555555555555554, 0.23450292397660819, 0.11345029239766082, 2.337311630789803e-17],
# [0.25029239766081873, 0.3380116959064327, 0.11871345029239766, 0.1888888888888889, -0.0005847953216373936, 0.39298245614035093],
# [0.20818713450292398, 0.39298245614035093, 0.21695906432748538, 0.1614035087719298, 0.37660818713450295, 0.22807017543859653],
# [0.38362573099415204, 0.35204678362573105, 0.19941520467836257, 0.34152046783625734, 0.16608187134502925, 0.22046783625731],
# [0.17309941520467836, 0.20116959064327486, 0.3719298245614035, 0.24912280701754388, 0.17192982456140352, 0.27894736842105267],
# [0.18070175438596492, 0.31695906432748544, 0.38362573099415204, 0.1625730994152047, 0.23976608187134504, 0.12631578947368424],
# [0.23801169590643279, 0.12280701754385968, 0.22923976608187135, 0.1625730994152047, 0.2701754385964913, 0.20350877192982458]]
# =============================================================================

# ==========================================PP===================================
# a = [[0.1932640706393811, 0.1138850129511065, 0.08686941330446639, 0.18169425402037878, 0.10843034481476094, 0.17494661967827568],
# [0.19116912683028942, 0.1800042628278159, 0.08438832346689383, 0.09861679591387185, 0.1940110104484737, 0.14921737733865317],
# [0.19434283814115882, 0.1809900216761659, 0.09030856705395045, 0.08937610655643094, 0.19586201883771967, 0.1946328196003503],
# [0.1052063810255068, 0.1896641954637843, 0.0933631721384483, 0.19489898653650345, 0.10302807080289285, 0.14045379790335494],
# [0.1998821111018711, 0.1957018844388094, 0.15765147782568087, 0.19571448472272907, 0.19339771653828286, 0.19341648855877414],
# [0.15746697375642132, 0.1660856029405294, 0.09799459884734352, 0.0965448750786128, 0.17514295693250295, 0.19575608945049458],
# [0.1737666747812635, 0.15390856779683276, 0.09862720152417749, 0.19318759590997955, 0.1806570649748373, 0.19412153362452533],
# [0.18622212877476962, 0.20137177663264655, 0.09443435555029749, 0.12394199509604784, 0.18275876467001076, 0.1939452310358155]]
# =============================================================================

a = [[0.2426076534058603, 0.2597354952733845, 0.16960767295739854, 0.13352035310090896, 0.25669111700205494, 0.2584724129765138],
[0.2567527764151615, 0.2361778724312181, 0.21744988232800688, 0.2603685024564862, 0.25980828192686295, 0.2505083487084716],
[0.2583590996797841, 0.2597649281724698, 0.1255953937355048, 0.15297926353439403, 0.24244879916281867, 0.2627821584332976],
[0.25939943584621167, 0.24414594937724826, 0.1302368246321732, 0.1477787440293231, 0.2541500414272286, 0.2588846014407025],
[0.21322566027825474, 0.2597135646378794, 0.16404126618112594, 0.16221463665559982, 0.25983670191517, 0.2416154990610521],
[0.25991247845112686, 0.26037268450763423, 0.12781860407625523, 0.12694547674311865, 0.20253708371768553, 0.2598319614339126],
[0.25992667270636666, 0.2599021033996871, 0.13964931731760497, 0.1277204044742385, 0.20514950908618992, 0.25051532168107266]]
A = np.array(a)
B = A.sum(axis=0)
print(B/7)