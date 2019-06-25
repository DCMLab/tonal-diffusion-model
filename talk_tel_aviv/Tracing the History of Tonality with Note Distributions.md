# Tracing the History of Tonality with Note Distributions

1. Tonality
   1. Notion, meanings, definition here (general construct)
2. Corpus
   1. Basic stats
3. Model 1: Bag of notes model
   1.  (multinomial)
4. Result 1: PCA fifths
5. Model 2: Tonal diffusion model
   1. motivation: tonal fields generalize "key" and thus capture a more abstract concept of tonality.
   2. tonal center is starting point from which tones "diffuse"
   3. [animation?] => machaut vs Schubert 90_2 (hex) vs Liszt Trauervorspiel (oct)
6. Result 2: 19th century, thirds
7. Conclusion



## 1 Tonality

* "Tonality" is a concept that is hard to define. Most generally, it describes the usage of notes in musical pieces and changes over time and between composers (also contemporaries)
* It may involve more specific concepts like "key", "mode", "scale", "root", "chord", "harmony"
* Some concepts like "key" and "mode" don't make sense in for any piece/time but refer to a specific type of tonality, the "common-practice tonality". The notion of "mode" is different when speaking of major and minor as opposed to the church modes
* But all of the mentioned concepts derive from more basic ones: from notes and their usage.
* Thus, the broadest possible definition of tonality is: the usage of notes. This usage is not random or arbitrary but follows certain conventions and rules that themselves define more narrow concepts of tonality from epoch-defining rules like manifest in counterpoint to composer-specific choices like Messiaen's modes.
* This abstract perspective allows to observe how the usage of notes (=Tonality) changes over time and analyze these observations in order to find patterns.
* These patterns define styles, dialects, and idioms. 

## 2 Corpus 

* Starting the Observation: the corpus
  * ~600 years
  * 75 composers
  * ~2,000 pieces
  * ~3,000,000 notes
* we treat all pieces the same (first model: bag of notes)
* this enables us to compare them and show differences and commonalities
* The most basic aspect of the usage of notes (=Tonality) is their frequency
* Counting notes is the most basic representation. But there is more than one way to count notes. One can count all note onsets, one can count how long each note lasts, one can adjust the weight of notes according to their metrical position, etc.
* No matter how we count, we get a bag of notes for each piece
* Since we are trying to derive a very general model of tonality, we are not imposing further assumptions on the relations between these nodes or, equivalently, we assume that they are all independent of each other. This is, of course, not a very musical assumption. We feel sure that we know that notes are not completely independent of each other but have indeed close relations. But remember that we want to find out something about tonality so we can not put these assumptions at the beginning.
* **Assumption 0:** Octave equivalence
* **Assumption 1:** Let's assume all notes in the bag are independent and that we have $V$ different notes that define our note vocabulary. Then, this defines a $V$-dimensional vector space and each piece points to a location in this space. 
* **Assumption 2:** Notes are uniformly distributed in this space.
  * Which test to perform for uniformity?! (chi square? KS?) => use $R^2$ as in ABC paper [how does "distance to uniformity" change over time? what is the Zipf fit for the overall tpc distribution?]
* Pieces van vary vastly in length (Figure?). As printed scores they can range from about half a page to a whole volume of hundreds of pages. 
* **Assumption 3:** The _relative_ not the absolute counts of notes define tonality. This assumption restricts our investigation to the $V$-dimensional simplex in piece space. 

## 3 Bag-of-notes model

* So, pieces can look like (Bach bminor) or (Liszt bminor) [just value_counts, sorted by freq]
* Since we have more than 3 notes in our vocabulary, it is impossible to visualize this space. But we know something about its properties: since it is a vector space, pieces that have similar note counts (bags) will be closer to each other than pieces that have very different note counts.
* Recall that we did not transpose the keys to a common root or something. One could conclude that, since untransposed similar pieces are close in the space, one could identify these clusters with keys. There are some problems with this: as mentioned before, the concept of key is not appropriate for any kind of music. What would it even mean if Parsifal would be close to, say, Bach's Ab major prelude, and Messiaen's 1st Prelude to Beethoven's E-major Sonata op. 109?
* Our goal is to derive more general principles of tonality and how it changes.
* Surely, the note distributions in the pieces vary greatly. But maybe they to so in a systematic way. Systematic variations (that are linear) can be found with so-called _Principle Component Analysis_ (PCA).
* For $v\leq V$,  PCA finds a $v$-dimensional hyperplane in the $V$-dimensional space such that the distance of all points to this hyperplane is minimized.

### 3.1 Line of fifths from piece distributions

* Applying PCA with $v=2$ to our corpus leads to an interesting finding: the data projected to the Euclidean plane shows a crescent-like pattern. This means that we can drop assumption 1 because clearly, many of the $V\gg 2$ dimensions are correlated. 
* But which notes are correlated with which? Since we dropped assumption 2, we know that notes are not distributed uniformly. We can conclude that it is on average possible to identify the most frequent note, which we call _tonal center_. 
* We can now assign a color to each note in the vocabulary and assign each piece the color of its tonal center. We do not pick the colors randomly but sort the vocabulary along the line of fifths
* [Figure] 
* Here, C is colored white, sharp notes are red, and flat notes are blue. Using this colormap we see that the crescent represents a bent segment of the line of fifths. This means that the note distributions in musical pieces vary mostly along the line of fifths. In other words: 
* the most frequent note of pieces that have similar note distributions are close on the line of fifths.
* Also: 
  * pieces in "keys" with fewer accidentals vary more (PC1); how is this related to the absolute distribution of notes?
  * sharps and flats vary from each other (rarely cooccur; PC2)
* SUMMARY. (also how does this relate to terms like key and scale)

### 3.2 Line of fifths from tonal-pitch class coevolution

## 5 Tonal diffusion model 

Assumptions:

* 3 primary intervals, 2 interval directions
* One Tonal center from which tones "diffuse" in one of the primary interval directions
* constant discount rate(s)

### 5.1 Approximating pitch-class distributions by tonal diffusion

Show three pieces [animations next to original dist] :

- purely fifth (Machaut), octa (Liszt Trauervorspiel), hexa (90_2)

### 5.2 Increasing importance of thirds in the 19th century

- absolute weights show dominance of fifth, both ascending and descending
- until ~1600 the ascending fifth decreases while the descending fifth decreases => shift from plagal to authentic?
- around 1700 (but a bit earlier: it's not Bach) increase in virtually all components 
- relative weights show drastic increase of thirds in 19th century: previously, the fifth dimensions were sufficient in order to explain the pitch-class distribution of pieces => drastic change in tonality that uses M3 and m3 dimensions that were previously practically non-existent.



## 6 Conclusion

Assumptions:

1. Octave equivalence
2. ~~notes are independent~~
3. ~~notes are uniformly distributed~~
4. relative note frequencies determine tonality

Results:

* Models can make implicit assumptions explicit and allow critical discourse
* The line of fifths is the primary space for tonal compositions across time
* the thirds axes of the Tonnetz become more relevant in the 19th century, pointing to a dramatic shift in tonality that is related to the emergence of Extended Tonality.