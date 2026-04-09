---
title: "1. L'Architecture Atomique "
weight: 1
---

# Section 1 : L'Architecture Atomique – Du Token à l'Attention

Bonjour à toutes et à tous ! J'espère que vous avez bu votre café, car nous allons aujourd'hui réaliser une véritable "IRM" du cerveau d'un LLM. Dans cette première section, nous allons ouvrir le fichier `model.py` du projet **minGPT**. 

> [!IMPORTANT]
**Je dois insister :** ne vous laissez pas tromper par la brièveté du code. Chaque ligne est le fruit de soixante ans de recherche en linguistique et en mathématiques. Nous allons voir comment Karpathy a traduit le mécanisme de l'attention — que nous avons étudié en Semaine 3 — dans un langage que votre GPU T4 peut comprendre. Respirez, nous entrons dans la forge atomique du sens.

---

## 1.1 L'Organisation du Dépôt : La beauté dans la frugalité

Le projet minGPT est devenu légendaire pour sa structure. Contrairement aux bibliothèques commerciales qui s'étendent sur des centaines de dossiers, Karpathy a tout condensé dans un seul répertoire : `mingpt/`. 

> [!NOTE]
**L'intuition de l'expert** : Un bon code pédagogique doit être "lisible d'une seule traite".

Le fichier central est `model.py`. Il contient environ 250 lignes. À l'intérieur, vous ne trouverez que trois classes majeures qui forment la hiérarchie du Transformer :
1.  **`CausalSelfAttention`** : L'unité de calcul (le "regard").
2.  **`Block`** : La matière grise (la réunion de l'attention et de la réflexion).
3.  **`GPT`** : L'édifice complet (le cerveau).

---

## 1.2 La Classe `GPT` : Le Squelette et sa Configuration

Tout commence par une structure de données simple mais puissante : la configuration. En Semaine 3.4, nous avons vu que les LLM sont définis par des hyperparamètres. Dans minGPT, cela se traduit par la classe `GPTConfig`.

```python
# [SOURCE: karpathy/minGPT/mingpt/model.py#L112]
@dataclass
class GPTConfig:
    block_size: int = 1024  # Taille de la fenêtre de contexte (Semaine 5.1)
    vocab_size: int = 50257 # Nombre de mots dans le dictionnaire (Semaine 2.3)
    n_layer: int = 12       # Nombre de blocs Transformer empilés
    n_head: int = 12        # Nombre de têtes d'attention (Semaine 3.1)
    n_embd: int = 768       # Dimension du vecteur de sens (Semaine 2.4)
```

> [!WARNING]
**Attention : erreur fréquente ici !** Beaucoup d'étudiants confondent `n_embd` et `vocab_size`. 
*   `vocab_size` est le nombre de mots que l'IA connaît (le dictionnaire).
*   `n_embd` est la précision de sa "pensée" (le nombre de coordonnées GPS pour chaque mot).

> [!IMPORTANT]
**Je dois insister :** la puissance du modèle dépend de l'équilibre entre ces chiffres. Si vous augmentez `n_layer`, vous augmentez la "profondeur" du raisonnement, mais vous saturez plus vite la mémoire de votre carte T4.

---

## 1.3 La Forge des Embeddings : Transformer le Verbe en Position

Rappelez-vous notre Semaine 2.4 : les mots doivent devenir des points dans l'espace. Karpathy implémente cela via deux couches fondamentales situées au tout début du modèle.

### A. L'Embedding de Token (`wte`)
Dans le code, vous trouverez : 
`self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)`

> **La traduction concrète** : C'est une table de correspondance géante. Quand le token n°1254 arrive, cette ligne va chercher la 1254ème ligne d'une matrice pour en extraire 768 nombres. C'est l'identité sémantique du mot. [SOURCE: karpathy/minGPT/mingpt/model.py#L140]

### B. L'Embedding de Position (`wpe`)
Comme nous l'avons appris en Section 3.2, le Transformer est "aveugle" à l'ordre des mots. 
`self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)`
Ici, Karpathy s'écarte du papier original (qui utilisait des ondes sinus/cosinus) pour utiliser des **embeddings positionnels apprenables**. 
*   Chaque position (de 0 à 1023) possède son propre vecteur que le modèle ajuste pendant l'entraînement.
*   **Le fusionnement** : Avant d'entrer dans les blocs, Karpathy fait une simple addition : `x = token_embeddings + position_embeddings`. 

> [!IMPORTANT]
On n'ajoute pas le *numéro* de la position, on ajoute un *vecteur* de position. C'est ce qui permet au modèle de comprendre que le mot à l'index 5 est physiquement proche de l'index 6.

---

## 1.4 La Cellule de Self-Attention : Le dialogue des matrices

Nous arrivons maintenant au "Saint des Saints" : la classe `CausalSelfAttention`. C'est ici que Karpathy code la formule de Vaswani que nous avons disséquée en Section 3.1.

```python
# [SOURCE: karpathy/minGPT/mingpt/model.py#L31]
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Le trio Query, Key, Value condensé en une seule opération matricielle
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # La projection de sortie (Semaine 3.3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
```

**L'astuce d'ingénieur de Karpathy** : Au lieu de créer trois couches séparées pour Q, K et V, il crée une seule couche trois fois plus large (`3 * n_embd`). C'est beaucoup plus efficace pour le GPU ! Une seule multiplication matricielle génère tout le matériel nécessaire pour l'attention.

### Le calcul de l'attention (Q, K, V en mouvement)
Dans la méthode `forward`, vous verrez le calcul suivant (simplifié pour l'explication) :
1.  **La Projection** : On sépare les résultats de `c_attn` pour obtenir nos trois matrices $Q, K, V$.
2.  **Le Score (Attention)** : `att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))`
    *   `q @ k` : C'est le produit scalaire (Dot Product) de la Section 3.1. On demande : "Qui s'intéresse à qui ?".
    *   `1.0 / math.sqrt(...)` : C'est le **Scaling** (Section 3.1). On calme les nombres pour éviter que le Softmax ne "sature".

---

## 1.5 Le Mécanisme de Causal Masking : L'interdiction de tricher

Mes chers étudiants, portez une attention toute particulière à ces quelques lignes de code. Elles sont la différence entre un modèle qui comprend et un modèle qui ne fait que réciter.

```python
# [SOURCE: karpathy/minGPT/mingpt/model.py#L46]
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
```


**Analyse de la [Figure 1-19]({{< relref "section-1-3.md" >}}#fig-1-19)** via le code : 
*   `torch.tril` crée une matrice triangulaire inférieure remplie de 1. 
*   Karpathy utilise ce masque pour dire au modèle : "Le mot en position 5 a le droit de regarder les mots 0, 1, 2, 3, 4 et 5. Mais il est **strictement interdit** de regarder les mots 6, 7, 8...". 
*   **Le mécanisme** : Avant le Softmax, on met une valeur de $-\infty$ (moins l'infini) sur les positions futures. Le Softmax transforme ce $-\infty$ en une probabilité de **0%**.

> [!IMPORTANT]
C'est ce que nous avons appelé le **Causal Masking** (Section 3.4). Sans ce masque, minGPT ne serait pas un modèle de génération, il serait un simple dictionnaire tricheur qui connaît déjà la fin de la phrase.

---

## 1.6 L'Intégration dans le Bloc Transformer

Enfin, Karpathy réunit tout cela dans la classe `Block`. C'est la mise en pratique de la [**Figure 3-10**]({{< relref "section-3-3.md" >}}#fig-3-10) .

```python
# [SOURCE: karpathy/minGPT/mingpt/model.py#L90]
def forward(self, x):
    x = x + self.attn(self.ln_1(x)) # Attention + Résidu (Semaine 3.3)
    x = x + self.mlp(self.ln_2(x))  # Feedforward + Résidu
    return x
```

> [!IMPORTANT]
**Je dois insister sur les Résidus (`x + ...`)** : Notez bien ce petit `x +`. C'est la connexion résiduelle. Sans elle, le signal se perdrait dans les 12 couches de minGPT et le modèle ne pourrait jamais apprendre. C'est l'autoroute du savoir.

---

## Synthèse

Vous venez de voir, ligne par ligne, comment les concepts de tokens, d'embeddings, de Query/Key/Value et de masquage causal s'assemblent pour créer un embryon d'intelligence. 

> [!TIP]
**Le message à retenir** : minGPT n'est pas "mini" parce qu'il est faible, il est "mini" parce qu'il ne garde que l'essentiel. En étudiant `model.py`, vous avez vu que l'intelligence artificielle n'est pas un mystère insondable, mais une **géométrie de l'attention**. 

---

Dans la **Section 2**, nous allons voir comment faire battre ce cœur. Nous allons étudier le `trainer.py` pour comprendre comment on injecte l'erreur et comment on ajuste ces milliards de petits boutons pour que le modèle finisse par parler comme un humain. Prêts pour l'ingénierie du gradient ?