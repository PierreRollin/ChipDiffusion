# ChipDiffusion : Arbitrage de Volatilité et Pricing Stochastique (Supply Chain Semi-conducteurs)

## 1. Fondations Mathématiques : Du Temps Discret au Temps Continu

Alors que les modèles prédictifs (LSTMs, HMMs) opèrent en temps discret pour anticiper les rendements futurs ($\mu$), l'évaluation de produits dérivés (Pricing d'options) exige un passage en temps continu. Sous la probabilité risque-neutre ($\mathbb{Q}$), l'espérance de rendement de l'actif devient non pertinente. La variable centrale du modèle devient la **Volatilité ($\sigma$)**.

### 1.1 Le Mouvement Brownien Géométrique (GBM)
Les prix des actifs ($S_t$) ne peuvent être modélisés par un simple Mouvement Brownien standard ($W_t$) car un prix ne peut être négatif, et la dynamique des rendements est multiplicative. Nous modélisons donc l'actif sous-jacent via l'Équation Différentielle Stochastique (EDS) du Mouvement Brownien Géométrique :

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Où :
*   $\mu S_t dt$ est la dérive déterministe (Drift).
*   $\sigma S_t dW_t$ est le choc stochastique, piloté par un processus de Wiener où $dW_t \sim \mathcal{N}(0, dt)$.

**Biais de Discrétisation :** Pour simuler informatiquement ce processus continu, j'ai implémenté le **schéma d'Euler-Maruyama** au sein d'un moteur Monte Carlo vectorisé. La convergence de l'espérance terminale simulée vers sa valeur théorique exacte $\mathbb{E}[S_T] = S_0 e^{\mu T}$ valide la robustesse de l'environnement stochastique.

### 1.2 Le Lemme d'Itô : L'Expansion de Taylor Stochastique
Pour évaluer un produit dérivé, tel qu'une option d'achat (Call) $V(S, t)$, il est impératif de comprendre sa dynamique par rapport à l'actif sous-jacent $S_t$. En calcul différentiel classique, la différentielle totale serait $dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS$. 

Cependant, en calcul stochastique, la variation quadratique du Mouvement Brownien est non nulle ($\mathbb{E}[(dW_t)^2] = dt$). Il est donc obligatoire de pousser le développement de Taylor au second ordre par rapport à $S$. C'est l'essence du **Lemme d'Itô** :

$$dV = \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \sigma S \frac{\partial V}{\partial S} dW_t$$

L'apparition de la dérivée seconde ($\frac{\partial^2 V}{\partial S^2}$, connue sous le nom de **Gamma**) démontre mathématiquement que la convexité et la variance ($\sigma^2$) génèrent intrinsèquement de la valeur dans le prix d'une option.
### 1.2.1 Application : La dynamique du Log-Prix et le mystère du $-\frac{1}{2}\sigma^2$

Pour comprendre le lien entre les données de marché discrètes et le modèle continu, partons du GBM :

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

L'objectif est de trouver une dynamique explicite pour $S_t$. Puisque $S_t$ agit comme un facteur multiplicatif, le réflexe analytique est de passer au logarithme. Posons la fonction $f(S) = \ln(S)$.

**1) L'échec du calcul classique**
Si $S_t$ était une fonction ordinaire et déterministe, la règle de la chaîne classique (Newton/Leibniz) donnerait :
$$d(\ln S_t) = \frac{1}{S_t} dS_t$$
En calcul stochastique, cette équation est fondamentalement fausse car elle omet la variation quadratique du Mouvement Brownien.

**2) La recette du Lemme d'Itô**
Si un processus suit $dX_t = a(X_t, t)dt + b(X_t, t)dW_t$, alors pour une fonction $f$ suffisamment régulière, la différentielle stochastique est :
$$df(X_t) = \left( \frac{\partial f}{\partial t} + a f' + \frac{1}{2} b^2 f'' \right)dt + b f' dW_t$$

Dans notre cas :
* $X_t = S_t$
* $a = \mu S_t$
* $b = \sigma S_t$
* $f(S) = \ln(S)$

**3) Calcul des dérivées et substitution**
* Première dérivée : $f'(S) = \frac{1}{S}$
* Deuxième dérivée : $f''(S) = -\frac{1}{S^2}$
* Dérivée temporelle : $\frac{\partial f}{\partial t} = 0$ (la fonction $\ln$ ne dépend pas explicitement du temps).

En substituant ces éléments dans la formule d'Itô :
* Terme en $dt$ de dérive temporelle : $(\mu S_t) \left(\frac{1}{S_t}\right) = \mu$
* Terme de correction quadratique : $\frac{1}{2} (\sigma S_t)^2 \left(-\frac{1}{S_t^2}\right) = -\frac{\sigma^2}{2}$
* Terme stochastique : $(\sigma S_t) \left(\frac{1}{S_t}\right) = \sigma$

L'équation finale de la dynamique du log-prix devient :
$$d(\ln S_t) = \left( \mu - \frac{\sigma^2}{2} \right)dt + \sigma dW_t$$

**4) L'origine mécanique de la correction (La preuve par Taylor)**
La question centrale est : pourquoi ce terme $-\frac{\sigma^2}{2}$ apparaît-il ?
Il provient du développement de Taylor à l'ordre 2 :
$$d(\ln S_t) \approx f'(S_t)dS_t + \frac{1}{2}f''(S_t)(dS_t)^2$$

Si l'on calcule le carré de l'incrément $(dS_t)^2$ :
$$(dS_t)^2 = (\mu S_t dt + \sigma S_t dW_t)^2$$
$$(dS_t)^2 = \mu^2 S_t^2 (dt)^2 + 2\mu\sigma S_t^2 (dt dW_t) + \sigma^2 S_t^2 (dW_t)^2$$

En algèbre d'Itô, les termes d'ordre supérieur à $dt$ convergent vers 0. Ainsi, $(dt)^2 = 0$ et $dt dW_t = 0$. En revanche, l'espérance de la variation quadratique du mouvement brownien est $\mathbb{E}[(dW_t)^2] = dt$. Il reste donc uniquement :
$$(dS_t)^2 = \sigma^2 S_t^2 dt$$

C'est l'injection de ce terme dans la dérivée seconde de Taylor qui génère la correction de volatilité. 

> **💡 Modèle mental :** 
> *Itô = Règle de la chaîne classique + Terme de correction dû au fait que $(dW_t)^2$ génère un incrément temporel déterministe ($dt$).* 
> Cette démonstration justifie mathématiquement pourquoi la moyenne des rendements logarithmiques observés sur le marché doit être corrigée du facteur $\frac{\sigma^2}{2}$ pour obtenir le vrai *Drift* de l'actif.

### 1.3 L'EDP de Black-Scholes-Merton (BSM) et le Delta-Hedging
L'innovation majeure de BSM repose sur le portefeuille de réplication. En construisant un portefeuille $\Pi$ composé de l'achat d'une option $V$ et de la vente à découvert de $\Delta = \frac{\partial V}{\partial S}$ actions du sous-jacent, le terme stochastique ($dW_t$) est parfaitement neutralisé (on calcul $d\Pi \ où\ \Pi=V-\Delta S$).

Ce portefeuille couvert en Delta étant localement sans risque, il doit impérativement rapporter le taux d'intérêt sans risque ($r$) afin de prévenir toute opportunité d'arbitrage ($d\Pi = r\Pi dt$). En égalisant la dérive déterministe de ce portefeuille avec le taux sans risque, on obtient l'**Équation aux Dérivées Partielles (EDP) de Black-Scholes** :

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

La résolution de cette EDP avec les conditions aux limites adéquates (ex: $V(S, T) = \max(S - K, 0)$ pour un Call Européen) fournit la formule fermée de Black-Scholes. Elle permet d'évaluer le juste prix de l'option instantanément, contournant la lourdeur algorithmique des simulations Monte Carlo, à la stricte condition que l'hypothèse de volatilité constante soit respectée.

### 1.4 Les Greeks : Métriques de Sensibilité et Gestion du Risque

Le prix d'une option $V$ n'est pas statique ; c'est une surface multidimensionnelle qui réagit aux perturbations de l'environnement (Prix du sous-jacent, Temps, Volatilité, Taux). L'évaluation de ces sensibilités se fait via le calcul des dérivées partielles de la formule de Black-Scholes, appelées communément "Les Grecques".

#### 1.4.1 Delta ($\Delta$) : L'Exposition Directionnelle
Le Delta mesure la sensibilité du prix de l'option par rapport à une variation infinitésimale du prix de l'actif sous-jacent ($S$).
$$\Delta = \frac{\partial V}{\partial S}$$
*   **Pour un Call :** $\Delta_{Call} = \mathcal{N}(d_1) \in [0, 1]$. Si l'action monte de 1\$, l'option Call monte de $\Delta\$$.
*   **Pour un Put :** $\Delta_{Put} = \mathcal{N}(d_1) - 1 \in [-1, 0]$.
*   *Application Quant :* C'est la métrique fondamentale du "Delta-Hedging". Un portefeuille Delta-Neutre ($\sum \Delta_i = 0$) est immunisé contre les petits mouvements directionnels du marché.

#### 1.4.2 Gamma ($\Gamma$) : La Convexité
Le Gamma est la dérivée seconde du prix de l'option par rapport au prix du sous-jacent. C'est le taux de variation du Delta.
$$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S} = \frac{\mathcal{N}'(d_1)}{S \sigma \sqrt{T}}$$
*(où $\mathcal{N}'$ est la densité de la loi normale).*
*   *Application Quant :* Le Gamma est maximal quand l'option est "At-the-Money" (ATM). Un Gamma positif signifie que le portefeuille gagne de l'argent de manière non-linéaire lors de mouvements violents (la volatilité est l'amie du Gamma). Une option deep ITM ou OTM est proche de 0. C'est le moteur de l'arbitrage stochastique.

#### 1.4.3 Vega ($\nu$) : L'Exposition à la Volatilité
Bien que ce ne soit pas une lettre grecque officielle, Vega mesure la sensibilité du prix de l'option face à un changement de la volatilité implicite ($\sigma$).
$$\nu = \frac{\partial V}{\partial \sigma} = S \sqrt{T} \mathcal{N}'(d_1)$$
*   *Application Quant :* Vega est strictement positif pour les Calls et les Puts. Dans notre stratégie `ChipDiffusion`, si le LSTM prédit une hausse massive de la volatilité sur NVDA, nous construirons un portefeuille avec un **Vega fortement positif** (achat d'options) tout en gardant un **Delta neutre** pour ignorer la direction du prix.

#### 1.4.4 Theta ($\Theta$) : La Décroissance Temporelle (Time Decay)
Theta mesure la perte de valeur de l'option à mesure que le temps passe (dérivée par rapport au temps $t$, ou inversement par rapport à l'échéance $T$).
$$\Theta = -\frac{\partial V}{\partial T} = -\frac{S \mathcal{N}'(d_1) \sigma}{2 \sqrt{T}} - r K e^{-rT} \mathcal{N}(d_2)$$
*   *Application Quant :* Theta est généralement négatif pour un acheteur d'option. C'est le "loyer" payé chaque jour pour bénéficier du Gamma. La relation fondamentale de Black-Scholes sans risque se résume par le compromis Gamma-Theta : on paie du Theta (temps) pour détenir du Gamma (convexité/mouvement).

### 1.5 Inversion du Modèle : La Volatilité Implicite et la méthode de Newton-Raphson

Dans la pratique des marchés financiers, le modèle de Black-Scholes n'est pas utilisé pour "prédire" le prix d'une option. Le prix est dicté par la loi de l'offre et de la demande dans le carnet d'ordres. L'inconnue de l'équation devient alors la volatilité $\sigma$. 

La **Volatilité Implicite (IV)** est la valeur de $\sigma$ qui, injectée dans l'équation de Black-Scholes, permet de retrouver exactement le prix observé sur le marché. Elle représente le consensus des acteurs du marché sur l'incertitude future.

L'équation de Black-Scholes n'étant pas inversible analytiquement (la variable $\sigma$ étant emprisonnée dans l'intégrale de la fonction de répartition $\mathcal{N}$), nous devons utiliser un algorithme d'optimisation numérique pour trouver la racine de la fonction :
$$f(\sigma) = Prix_{BSM}(\sigma) - Prix_{Marché} = 0$$

**L'algorithme de Newton-Raphson :**
Grâce au calcul des Grecques, nous possédons la dérivée exacte du prix par rapport à la volatilité : le **Vega** ($\nu$). Le Vega est le gradient parfait pour notre descente algorithmique.
La mise à jour itérative de la volatilité s'écrit donc :
$$\sigma_{n+1} = \sigma_n - \frac{Prix_{BSM}(\sigma_n) - Prix_{Marché}}{\nu(\sigma_n)}$$

L'algorithme converge généralement en moins de 5 itérations vers la volatilité implicite avec une précision de $10^{-5}$, permettant d'extraire le "niveau de peur" pour chaque actif de la chaîne des semi-conducteurs.

## 2. Le Choc de Réalité Empirique : La Surface de Volatilité

L'un des postulats fondateurs du modèle de Black-Scholes est que la volatilité ($\sigma$) du sous-jacent est constante. Par conséquent, toutes les options sur un même actif devraient afficher la même Volatilité Implicite (IV), quels que soient leur Strike ($K$) ou leur Maturité ($T$).

### 2.1 Volatility Skew et "Fat Tails"
L'observation empirique des marchés (particulièrement depuis le krach d'octobre 1987) invalide cette hypothèse. Les rendements boursiers ne suivent pas une loi log-normale parfaite ; ils présentent une asymétrie négative (Skewness) et des queues de distribution épaisses (Leptokurticité). 
En pratique, les acteurs du marché sont prêts à surpayer les options *Out-of-the-Money* (OTM), notamment les Puts, pour se protéger contre des krachs. Cette "prime de peur" courbe la ligne de volatilité implicite, formant un **Volatility Skew** (ou Smile).

### 2.2 La Nappe de Volatilité (Volatility Surface)
En étendant cette analyse sur deux dimensions (Axe X : Moneyness $K/S_0$, Axe Y : Maturité $T$), l'IV forme une surface tridimensionnelle. Cette nappe est la représentation topographique de la peur et de l'incertitude. L'extraction de cette nappe sur des actifs comme NVIDIA ou SMIC permet de cartographier la structure par terme du risque. 
*Note d'ingénierie : Cette extraction est bruitée par le manque de liquidité des options profondes (Deep ITM/OTM), nécessitant un filtrage strict par le volume et l'écart Bid-Ask pour garantir la convergence de l'algorithme de Newton-Raphson.*

---

## 3. Mathématiques de l'Arbitrage de Volatilité (Volatility Risk Premium)

C'est l'écart entre la théorie de Black-Scholes et la réalité du marché qui génère l'opportunité d'arbitrage. Le but n'est pas de parier sur la direction de l'action, mais sur l'erreur d'estimation de sa volatilité par le marché.

### 3.1 Volatilité Implicite (IV) vs Volatilité Réalisée (RV)
*   **La Volatilité Réalisée (RV)** est l'écart-type historique empirique des rendements de l'actif. C'est ce que l'action *fait réellement*.
*   **La Volatilité Implicite (IV)** est la volatilité "pricelée" par le marché. C'est ce que le marché *anticipe*.

Structurellement, il est démontré que $\text{IV} > \text{RV}$ la majorité du temps. Cette différence s'appelle la **Volatility Risk Premium (VRP)**. C'est une prime d'assurance payée par les acheteurs d'options aux "Market Makers" pour le transfert du risque.

### 3.2 L'Équation Fondamentale du PnL (Gamma-Theta Trade-off)
Comment monétiser la différence entre IV et RV sans s'exposer aux variations de prix du sous-jacent ? La solution réside dans le **Delta-Hedging** dynamique.
Considérons un portefeuille $\Pi$ contenant une option vendue, couverte en permanence par l'achat de $\Delta$ actions. En appliquant le lemme d'Itô sur ce portefeuille, la variation du Profit & Loss (PnL) sur un instant $dt$ ne dépend plus du rendement de l'action, mais uniquement de sa variance :

$$dPnL \approx \frac{1}{2} \Gamma S^2 (\sigma_{imp}^2 - \sigma_{real}^2) dt$$

Où :
*   $\Gamma$ est le Gamma de l'option (sa convexité).
*   $\sigma_{imp}$ est la volatilité implicite à laquelle l'option a été tradée.
*   $\sigma_{real}$ est la volatilité qui se matérialise réellement sur l'intervalle $dt$.

**Conclusion stochastique :** Si un algorithme prédictif (ex: LSTM + HMM) parvient à anticiper que la volatilité qui va se réaliser ($\sigma_{real}$) sera significativement inférieure à la volatilité actuellement "pricelée" par le marché ($\sigma_{imp}$), il est mathématiquement prouvé qu'un portefeuille Delta-Neutre générera un profit certain, indépendamment de la hausse ou de la baisse de l'action.

### 3.3 L'Instrument Stratégique : Le Straddle ATM
Pour maximiser l'exposition à la volatilité tout en annulant l'exposition directionnelle initiale, l'instrument roi est le **Straddle At-The-Money (ATM)**.
Il consiste à acheter (ou vendre) simultanément un Call et un Put ayant le même Strike $K \approx S_0$ et la même maturité $T$.
*   $\Delta_{Straddle} = \Delta_{Call} + \Delta_{Put} \approx 0.5 + (-0.5) = 0$ (Risque directionnel nul).
*   $\nu_{Straddle} = \nu_{Call} + \nu_{Put}$ (Exposition maximale à la Volatilité, le Vega étant à son apogée à la monnaie).