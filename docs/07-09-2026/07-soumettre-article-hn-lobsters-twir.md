# Soumettre l'article — Hacker News, lobste.rs, This Week in Rust, dev.to, r/programming (7 septembre 2026)

L'article : `https://l-defraiteur.github.io/lucivy/blog/every-engine-lies-a-little.html`
(page canonique, carte de lien `og-card.png`), copie markdown dans `06-…md`.
Il parle de la mesure, pas du produit : c'est ce qui le rend acceptable là où
un post « mon projet » est refusé (r/programming) ou mal vu (r/rust).

Ce que r/rust a appris aujourd'hui : un post à la forme trop rangée (gras en
tête de paragraphe, tableau, phrases parallèles) est lu comme généré et
downvoté avant lecture. L'article est écrit à la première personne, avec des
phrases inégales et une opinion ; ne pas le « nettoyer ».

## Hacker News — le canal qui compte

- **Compte** : news.ycombinator.com, un compte à ton nom, même neuf. Pas de
  karma requis pour soumettre.
- **Soumettre** : news.ycombinator.com/submit, champ *title* + *url*. Rien
  d'autre : pas de texte quand il y a une URL.
- **Deux soumissions possibles, pas le même jour** :
  1. **L'article**, titre tel quel, sans préfixe : `Every full-text engine lies a little – measured on the Linux kernel`. HN retire un « I », « my », les majuscules de titre ; garder le titre de la page, c'est la règle.
  2. **Le projet**, plus tard, en `Show HN: Lucivy – substring, fuzzy and regex search with every answer checked against the files` (URL : la page du playground, où l'on peut *essayer* — c'est la condition d'un Show HN). Dans le premier commentaire, trois lignes : ce que c'est, le prix (la taille), ce qu'on aimerait qu'on critique.
- **Quand** : un jour de semaine, entre 14 h et 17 h heure de Paris (le matin
  sur la côte est). Le week-end est plus calme mais les gens lisent plus
  longtemps ; éviter le vendredi soir.
- **Après** : répondre à chaque commentaire technique dans l'heure, avec les
  chiffres, en concédant ce qui est vrai. Ne jamais demander de votes (leur
  détecteur d'anneau de votes bannit le post). Si le post ne prend pas, on a
  le droit de le resoumettre **une** fois quelques jours plus tard, à une
  autre heure ; les modérateurs le font parfois d'eux-mêmes (« second-chance
  pool ») pour un bon article passé inaperçu — un mail à hn@ycombinator.com
  pour le demander est accepté.

## lobste.rs

- **Sur invitation seulement.** Il faut qu'un utilisateur existant t'invite
  (lobste.rs/invitations ; le chat IRC `#lobsters` sur Libera accepte les
  demandes avec un lien vers ce qu'on a publié). Sans invitation, quelqu'un
  d'autre peut soumettre l'article — c'est courant et accepté.
- **Soumettre** : lobste.rs/stories/new, URL + titre + tags `rust`,
  `search`, `performance` ; cocher « I am the author » (l'étiquette *authored
  by* est bien vue, pas l'inverse).
- Audience petite (quelques milliers) mais qui lit et commente sérieusement ;
  souvent repris par HN ensuite.

## This Week in Rust

- Le lundi est bouclé le mardi soir ; les liens se proposent par **pull
  request** sur `rust-lang/this-week-in-rust`, dans le brouillon
  `draft/YYYY-MM-DD-this-week-in-rust.md`, section « Rust Walkthroughs » ou
  « Project/Tooling Updates » selon le contenu : l'article va dans
  *Walkthroughs* (c'est une mesure expliquée), la 4.0 dans *Project Updates*.
  Format : `* [Titre](url)` ; PR courte, titre « Add lucivy article ».
- **Crate of the Week** : nomination dans le fil hebdomadaire « Crate of the
  Week » sur r/rust (un commentaire : « I'd like to nominate lucivy-core… »),
  ou via l'issue épinglée du dépôt TWiR. Ça, c'est une auto-nomination qui
  est acceptée.

## dev.to

- Créer le compte, « Write a post », coller la copie markdown telle quelle,
  **canonical_url** = la page du site (dans les réglages du post, sinon
  Google voit un doublon). Tags `rust`, `search`, `performance`, `webdev`.
- dev.to est accepté comme source sur r/programming et HN.

## r/programming

- Pas de projet, mais un **article** oui : lien vers la page (ou dev.to),
  titre = le titre de l'article, sans flair. Un seul lien, pas de texte.
- Attendre deux jours après r/rust ; ne pas poster le même jour sur HN et
  r/programming (les deux se cannibalisent).

## Ordre proposé

1. Aujourd'hui : dev.to (canonique posé), puis rien.
2. Demain 14 h-17 h : HN, l'article.
3. Mardi : PR This Week in Rust (article + 4.0 dans Project Updates), r/programming.
4. Jeudi ou la semaine suivante : Show HN du playground, avec le premier
   commentaire prêt.
5. lobste.rs quand une invitation existe, ou dès que quelqu'un l'y a mis.

## Ce que je prépare si tu veux

Le premier commentaire du Show HN ; les réponses aux objections prévisibles
(« pourquoi pas tantivy », « 6× le texte c'est trop », « et Zoekt ? »,
« les trigrammes + vérification font pareil ») avec les chiffres du rapport,
à relire et coller.
