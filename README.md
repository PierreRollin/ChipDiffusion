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