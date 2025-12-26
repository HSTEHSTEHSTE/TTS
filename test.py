from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import os
from pathlib import Path
from tqdm import tqdm

config = XttsConfig()
config.load_json("/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
checkpoint_dir = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2'
# checkpoint_dir = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_base_mono-May-15-2025_09+47AM-e355c13'
model.load_checkpoint(config, checkpoint_dir = checkpoint_dir, eval = True)
print(checkpoint_dir)

model.cuda()
# out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/xtts_ul'
# out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/edacc/xtts_mixed'
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/cv_inference/eval/base_mono'
print(out_path)

# speaker_wavs = [
#     "/home/hltcoe/xli/ARTS/Recording.wav",
#     "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/84/121123/84-121123-0002.flac",
#     "/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id09185/_cDhsiYaV3c/00111.wav",
#     "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/251/118436/251-118436-0002.flac"
# ]
# speaker_wavs = [
#     "/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/data/eval_segments/spks/EDACC-C06-000000033.wav",
#     "/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/data/eval_segments/spks/EDACC-C09-000000084.wav",
#     "/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/data/eval_segments/spks/EDACC-C35_P2-000000133.wav",
#     "/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/data/eval_segments/spks/EDACC-C63-000000147.wav"
# ]
speaker_wavs = [
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/0.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/1.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/2.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/3.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/4.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/5.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/6.mp3",
    "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/7.mp3",
]
# speaker_wavs = [
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/8.wav",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/9.flac",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/10.wav",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/11.flac",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/12.flac",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/13.flac",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/14.flac",
#     "/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/15.flac",
# ]
gpt_cond_len = 3
language = "en"


# test_sentences = [
#     # 1-10
#     "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
#     "This cake is great. It's so delicious and moist.",
#     "What is the most resilient parasite? Bacteria? Virus? Intestinal Worm?",
#     "Two peanuts were walking down the road. One was assaulted. Peanut.",
#     "What is the airspeed velocity of an unladen swallow?",
#     "No human knows what everyone thinks, but such is life. Was that philosophical enough?",
#     "Cease, cows, life is short.",
#     "Do you want to take a leap of faith, or become an old man, filled with regret, waiting to die alone?",
#     "I've spent my life avoiding love and romance, they are distractions.",
#     "I say the whole world must learn of our peaceful ways, by force!",
#     #11-20
#     "I don't believe it is sign of strength to keep moving forward no matter what.",
#     "What makes a species turn neutral: lust for gold? Power? Or were they just born with hearts full of neutrality?",
#     "Gravity is desire, time is sight.",
#     "What was, will be; what will be, was.",
#     "Finally, I am very familiar with distributed GPU computing, having had a stint at NVIDIA developing CUDA kernels.",
#     "AI research is a collective effort to larp as scientists.",
#     "Once upon a time we also thought we were making history.",
#     "If our struggle was hopeless their propaganda would be unnecessary. Keep going.",
#     "My favorite activity is kicking asses, I spend most of my free time getting my ass kicked.",
#     "Hallucinations are only good for discovery if the model happens to have an underlying distribution that corresponds to the hyperplane of possibilities.",
#     #21-30
#     "For a quick recap, the goal of the Voice Privacy Challenge is to remove as much of the speaker-identifying information from each utterance as possible.",
#     "As wise men are astonished at foolish things, and other people at wise ones, I know not on which ground to account for Mr. Burke's astonishment.",
#     "Knowing you have to justify your answer makes you give a different answer.",
#     "The perfect spoofing detector is the perfect spoof.",
#     "Most of us have the distinct pleasure of going throughout our lives bereft of the physical presence of those who rule over us.",
#     "Were we peasants instead of spreadsheet jockeys, warehouse workers, and baristas, we would toil in our fields in the shadow of some overbearing castle from which the lord or his steward would ride down on his thunderous charger demanding our fealty and our tithes.",
#     "To their great advantage, they can buy their way out of public life.",
#     "However, if you want to catch a glimpse of them, all you need to do is attend a single day of Formula 1 racing.",
#     "The world of other people's sports is always an interesting one to the journalist who tends to stick to a single beat.",
#     "My sports work is primarily in cycling, though I have covered NASCAR in these very pages.",
#     #31-40
#     "There is certainly no need to apologise for writing in English.",
#     "By contrast, never once before have I attempted to write in English outside of professional or academic contexts.",
#     "If there is one thing that I still can't help complaining about though, it would be the rampant consumerism here in the US.",
#     "It is perhaps reflected in how people take towards me as well: rarely do I ever come across someone who is full of themselves the way many at Oxford could be perceived as.",
#     "The fact that most people I come across are immigrants or children of immigrants makes integration almost seemless, something I would have never expected in the UK.",
#     "Sorry for taking so long to write back to you - I still struggle to convince myself that I have adequated addressed all of the points you made in your letter.",
#     "My current work is in the intersection between security and speech technologies.",
#     "Poisoning attacks are an attack vector that has gained increasing attention in the adversarial machine learning research community.",
#     "Refinements to poisoning attacks, including 'clean label attacks' where a human would consider injected samples as correctly labeled, further complicates defenses against such types of attacks.",
#     "Knowing Lovecraft, lovecraftian horros were probably just about minorities.",
#     #41-50
#     "We just keep saying words whose meaning we do not know.",
#     "You want to grab it by the scruff, but it sieves away through your hands like sand; you want to sit and let it come, but it just goes in circles.",
#     "Too much abstraction and you lose grasp of reality; too little, and you get caught up in the details.",
#     "Like a spinning wheel, going ever faster and flashing patterns.",
#     "It keeps going faster, to the point where your eyes can no longer register but the most prominent ones, leaving the rest sitting in the background, in the subconscious mind.",
#     "We are, each of us, gods, creating worlds in our own image, and destroying worlds at our own will.",
#     "We are either completely alone in this universe or we are not. Both are extremely funny.",
#     "Life either has meaning, or it doesn't. Both are extremely funny.",
#     "As Matthew mentioned just now, one of the weaknesses of the cascaded speech translation approach is the lack of uncertainty propagation.​",
#     "That said, we can't simply propagate the embeddings generated by an ASR system to the downstream MT system due to their inherent differences. ​",
#     #51-60
#     "Unfortunately, matching the embeddings for ASR systems and MT systems comes at a cost. ​",
#     "For one, ASR and MT systems typically use very different vocabulary sizes.",
#     "Moreover, forcing an MT system to use a sub-optimal, almost semantically nonsensical embedding layer could potentially harm MT performance to such an extent that any potential downstream benefits cannot make up for that initial loss.​",
#     "In this track, we were tasked with translating recordings of ACL paper presentations into 10 different languages.",
#     "We focused primarily on cascaded speech translation approaches.",
#     "We use the following procedure to compute Technical Term Recall on our ASR output.",
#     "First we invited domain experts to annotate the reference transcript, highlighting any word that they would consider specific to the NLP domain.",
#     "Shanghainese is actually a pitch accent language.",
#     "Exploit the unique nature of sequence-to-sequence models.",
#     "What kind of mathematical properties are we trying to establish with manifolds?",
#     #61-70
#     "Find and finalise qualifying exam board to be ready for exam at the beginning of next semester.",
#     "The sprawl, with its never-ending blocks of shiny skyscrapers, rather than uprooting and urbanizing the town like it did the rest of the city, instead wrapped around Jiangwan and enveloped its crumbling old buildings and abandoned rail tracks.",
#     "For the next ten minutes, I will give an overview of the language and modality-disentangling decoder that we worked on during this workshop.",
#     "Subsequently, the decoder needs to map embeddings from this common space onto text or speech in the correct language and modality.",
#     "There are a number of possible approaches that could be taken to achieve this.",
#     "As such, we introduce the pipeline for our decoder as shown here.",
#     "Next we will discuss the architecture of our modality-disentangling decoder.",
#     "The decoder consists of two main components: the transformer decoder and the optional quantization layer.",
#     "The transformer decoder takes the output of our encoder as memory.",
#     "The target embeddings are used as target, with prompts prepended to the target.",
#     #71-80
#     "The quantization layer is introduced because our vocoder takes a finite set of discrete tokens as input.",
#     "While this discretisation could be performed on the output through methods such as K-means clustering, learning it directly in the decoder should give us performance benefits.",
#     "We base our first approach on the sequence length control system that was proposed in the Tacotron 2 paper.",
#     "The termination gate is trained against a sequence which has value 0 for the length of the sequence, and value 1 from end-of-sequence onwards.",
#     "One weakness in this approach is that the decision to terminate is made locally, without considering the whole sequence.",
#     "Thank you for guiding me into the most fascinating world of speech research! ",
#     "Thank you for the words of wisdom you gave me during my most hesitant moments, for sharing the insights that I needed to keep my anxieties about the future in check.",
#     "The first philosophers were astronomers.",
#     "The heavens remind man of his destination, remind him that he is destined not merely to act, but also to contemplate.",
#     "Fatalism is the most dangerous thing in the world.",
#     #81-90
#     "If you torture the data long enough, it will confess to anything",
#     "Language defines humanity, and language embodies humanity.",
#     "To study language without appealing to human nature, one risks fetishising it.",
#     "I see language not so much as Hegel's god, but more as Feuerbach's god: fundamentally a human creation, while also strictly a reflection of humanity, of human nature.",
#     "The more you look at language the more you understand about the world; the more you look into language, the more you understand about yourself.",
#     "I would also like to reference Hume's thesis on the relationship between rationalism and emotions.",
#     "Language at its core is an emotive tool.",
#     "When we come up with precise rules to define language, we are merely justifying our passions with the guise of rationality.",
#     "I like a place if I like the clouds in its sky.",
#     "I’m not sure which is worse: ignorant partisanship or calculated neutrality.",
#     #91-100
#     "Unlike many cultured families who tend to invoke references to traditional Chinese literature or mythology when naming their kids, my family was made up of humble folks who gave me a rather simple name.",
#     "Somewhere down the line I found myself training in maths and computer science and getting pretty good at it.",
#     "Unfortunately, those aren't exactly revealing the secrets to life and the universe. Or are they?",
#     "What does it take to present at reading group?",
#     "Research is generative, exams are discriminative.",
#     "Sophisticated engineers howling at each other, getting assaulted by a beetle.",
#     "A model of a shit emoji appears, inspiring the engineers.",
#     "The engineers then notice an F1 car model and start tossing it at each other.",
#     "The strong do what they can and the weak suffer what they must.",
#     "The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as what young Arthur wanted."
# ]


# test_sentences = [
#     # C06-A
#     "is that uh i mean realistically i would rather go to the beach today",
#     "i mean obviously but it's been i'd rather go right now as well as everybody is going there and a little bit grim",
#     "ooh okay tell me about a particular food you used to like when you were a child then",
#     # C09-B
#     "so what are we talking about",
#     "uh for me i think it's harry potter like",
#     "what i had imagined from the book",
#     # C35-A
#     "hi my name my participant's participant's number is f c two seven dash p one",
#     "no but it's like the it's the city in lithuania",
#     "women young child like eh holding guns they're soldiers how you can't remember that scene that part",
#     # C63-A
#     "and like then like stuff is already like all this other stuff's already like oh well that's too bombed well now there's like new information coming out like who knows and maybe there is something going on",
#     "i'm not sure because obviously we don't know the ending how like realistic that is",
#     "but i think he could probably turn out another one honestly",
# ]
# test_sentences = [
#     # US
#     "Many different routes have been travelled through the years.",
#     "Pasturella can be transmitted through the bite of a dog.",
#     # England
#     "The development of these forces took place in three stages.",
#     "She enjoys visiting Florida, because of the climate.",
#     # India
#     "He was killed while trying prevent illegal land grabbing and logging.",
#     "Don't you see what a wonderful thing this can be?",
#     # Africa
#     "My niece sent me a nice photo of Moscow via email.",
#     "Instead, Africa, its music, land, people, spirituality, tie us all together as a planet."
# ]


# test_sentences = [ # Philippines
#     "Such an injury requires surgical correction.",
#     "Everyone else has.",
#     "Young boy running outside on the pavement.",
#     "The track 'Remedy' was featured in the soundtrack for Tony Hawk's Underground.",
#     "Their presence was necessary because of strong civil unrest in the area.",
#     "The township includes the City of Saint Martin.",
#     "The Ancients were also called 'Soarers' for their common appearance in the air.",
#     "Muzzled greyhounds are racing along a dog track.",
#     "The current president is Pavel Cebanu.",
#     "Later they were thought to be deposits of volcanic ash or streaks of dust.",
#     "Price identifies as gay.",
#     "It is also funded through the National Lottery, Creative Scotland and Northern Ireland Screen.",
#     "The present comprehensive school is run by the Penelakut.",
#     "Her intro later appeared on the David Holmes Essential Collection.",
#     "Enemies killed by beam attack won't drop any orbs.",
#     "Peredur avenges his family and is celebrated as a hero.",
#     "In all cases the horse was adopted into their culture and herds multiplied.",
#     "The leaves are succubous.",
#     "Leclercq instead opted to record live choirs and solo gospel singers.",
#     "Chile is today one of South America's most stable and prosperous nations.",
#     "He also translated the Odes of Anacreon.",
#     "Prosperity is ascribed to their favor, and misfortune to their anger.",
#     "The album eventually sold over four million records."
# ]

# test_sentences = [ # Scotland
#     "Problems with the new treaty soon arose.",
#     "Tilak's son, Devdatt Narayan Tilak, edited and published the epic poem Christayana.",
#     "The spotted coat is caused by a genetic mechanism called the Leopard complex.",
#     "The Spiral ... stands for imagination, the power of an idea.",
#     "It was at University where Hamilton became politically active.",
#     "Flowering occurs between April and June.",
#     "Let me see your ticket.",
#     "The city has a high-desert climate that averages in January and in July.",
#     "This added the Alexandra Dock and Bootle Balliol Road stations to the line.",
#     "I don't like superstars with Botox faces.",
#     "He also directed several episodes of Barney Miller as Maxwell Gail.",
#     "Mark Kemp of Paste dismisses the song as a meandering bore.",
#     "Each night I pray they will arrest a bricklayer and a plumber.",
#     "Ren Ozawa was removed from the cast following his domestic abuse allegations.",
#     "Bergonzi began his career in the Atalanta Youth Sector.",
#     "Before being removed from the soil to represent the goddess, the plants are worshipped.",
#     "However, in the subway, he is approached by Higgins.",
#     "They were known as seers, and they were held in fear by women and the elderly.",
#     "Both nations are members of the United Nations.",
#     "It is similar to the Renderman Shading Language.",
#     "Must have been a kid.",
#     "In practice, however, other laws made it virtually impossible.",
#     "It was an unusually solid construction, capable of holding several hundred people.",
#     "This unusual sight often subdued the Chinese onlookers.",
#     "Riley was found by police, dead from pneumonia, in his basement hideout in Chinatown.",
#     "News, Vic Grimes and Redd Dogg.",
#     "He is also capable of firing lightning bolts with immense destructive power.",
#     "If all nodes have transmitters of equal power, these circles are all equal.",
#     "Minibikes are not toys, although often treated as such.",
#     "The odor of spring makes young hearts jump."
# ]

# test_sentences = [ # US
#     "Double seaming uses rollers to shape the can, lid and the final double seam.",
#     "His high school is unknown.",
#     "Because of a lack of parliamentary time, it never became law.",
#     "Four women competing in beach volleyball beside a large body of water.",
#     "Some theorists believe this is how the butter creation process was discovered.",
#     "Usually, speeds are stated in words per minute.",
#     "McDowell also cracked some ribs filming the humiliation stage show.",
#     "The twenty-eigtheen Olympics will be in Pyeongchang.",
#     "Hickie has suffered many injuries throughout his career.",
#     "It is distinct from vestments in that it is not reserved specifically for services.",
#     "This add-on allowed Notes documents to be rendered as web pages in real time.",
#     "The track, an alternate version of Kix is called Cha-cha-cha, sung like Elvis Presley.",
#     "They are all the family she has now.",
#     "Please do not feed the ducks!",
#     "When he insists on returning for his sister, Rachel relents and goes with him.",
#     "It was my intention to have a word with Angela.",
#     "It seemed young to me to demand the oath from.",
#     "She works hard, very hard.",
#     "Turkey is dismantling the incentive system.",
#     "What more do you want?",
#     "The majority of the residents of the Caroline area commute to the Ithaca area.",
#     "Well-funded public pensions have nearly wiped out poverty among the elderly.",
#     "Like its predecessor The Marble Index, it is an avant-garde album with neoclassical elements.",
#     "He did not finish.",
#     "The penguin waddles across the ice, joining her companions.",
#     "Always close the barn door tight.",
#     "He joined the United States Air Force straight out of high school.",
#     "It is lying crossways to the carrier.",
#     "Hello, my name is Brenda.",
#     "In that same broadcast, she referred to LaBarbera as her good friend."
# ]

test_sentences = [ # Australia
    "Underside dark ochraceous red.",
    "Since the horses roam the island, visitors may have to search for them.",
    "I played this for my brother and he knew almost all of them.",
    "These row vectors form the Gale diagram of the polytope.",
    "If you see something you like, take it and make it better.",
    "How they managed to improvise at such short notice, she'll never know.",
    "Mayday, we crashed into an iceberg!",
    "We must then compute the saturation pressure.",
    "She is working in a small restaurant and dreams of becoming an actress.",
    "In April of that year the girls' official fan club was renamed Hello!",
    "One of North Carolina's state aquariums is located here.",
    "He was a member of the Constituent Assembly and of the Executive Commission.",
    "The Minister did not reply to this letter.",
    "It was speculated that the engine which powered the officers kayak was heard.",
    "Confusion about his first place of burial has arisen from this connection to Lilleshall.",
    "The five locks at Knowle raise the canal to its summit.",
    "The fruits are ovoid and measure up to long.",
    "Did you really need to clog up that pipe?",
    "Kovu returns to Pride Rock to plead Simba for his forgiveness but is exiled.",
    "The main town in the region is Fagernes, where there also is an airport.",
    "Wonderland is now the site of a business park called Interchange Park.",
    "Do you have any suspicions as to who they might be?",
    "Yarmouk University has a newspaper that published periodically.",
    "Antonio was Ambassador of Portugal in Doha.",
    "Mr Lincoln is my mentor.",
    "Donald Brydon was selected to be its chief executive officer.",
    "Allied air forces included units of the Royal New Zealand Air Force.",
    "Can I have some of your tasty beverage?",
    "They said: Be not in awe.",
    "The firm represented Martha Stewart during her six-month insider trading trial."
]


# test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa']
# test_accents = ['England', 'US', 'India', 'Southern Africa']
# test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa', 'Canada', 'Australia', 'Philippines', 'Scotland', 'Ireland', 'Malaysia', 'Wales']
test_accents = ['Australia']
test_langs = ['en', 'en', 'hi', 'de', 'ar', 'en', 'en', 'hi', 'en', 'en', 'hi', 'en']

print(test_accents)
for test_accent_index, test_accent in enumerate(test_accents):
    target_path = os.path.join(out_path, test_accent)
    Path(target_path).mkdir(parents = True, exist_ok = True)
    for sentence_index, sentence in tqdm(enumerate(test_sentences), total = len(test_sentences)):
        for target_speaker_index, speaker_wav in enumerate(speaker_wavs):
            out_wav_path = os.path.join(target_path, 'sentence_' + str(sentence_index) + '_spk_' + str(target_speaker_index) + '.wav')
            # outputs = model.synthesize(
            #     sentence,
            #     config,
            #     speaker_wav = speaker_wav,
            #     gpt_cond_len = gpt_cond_len,
            #     language = 'en',
            #     accents = test_accent
            # )
            outputs = model.synthesize(
                sentence,
                config,
                speaker_wav = speaker_wav,
                gpt_cond_len = gpt_cond_len,
                # language = test_langs[test_accent_index],
                language = 'en',
            )
            torchaudio.save(out_wav_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
