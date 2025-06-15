// Data files would be in the same directory
const QUESTIONS_URL = 'questions.json';
const QUOTES_URL = 'quotes.json';
const VIDEOS_URL = 'videos.json';

// DOM Elements
const stage1 = document.getElementById('stage1');
const stage2 = document.getElementById('stage2');
const stage3 = document.getElementById('stage3');
const quizForm = document.getElementById('quiz-form');
const quizQuestion = document.getElementById('quiz-question');
const quizOptions = document.getElementById('quiz-options');
const quoteText = document.getElementById('quote-text');
const quoteAuthor = document.getElementById('quote-author');
const nextBtn = document.getElementById('next-btn');
const videoBtn = document.getElementById('video-btn');
const videoTitle = document.getElementById('video-title');
const videoModal = document.getElementById('video-modal');
const videoEmbed = document.getElementById('video-embed');
const closeBtn = document.querySelector('.close-btn');
const restartBtn = document.getElementById('restart-btn');

// App state
let currentStage = 1;
let questions = [];
let quotes = [];
let videos = [];
let dayIndex = 0;
let firstVisitDate = null;

// Initialize the app
async function init() {
    // Load first visit date from localStorage or set it
    const storedDate = localStorage.getItem('firstVisitDate');
    if (!storedDate) {
        firstVisitDate = Date.now();
        localStorage.setItem('firstVisitDate', firstVisitDate.toString());
    } else {
        firstVisitDate = parseInt(storedDate);
    }

    // Calculate day index
    const today = new Date();
    const firstDate = new Date(firstVisitDate);
    const timeDiff = today - firstDate;
    const dayDiff = Math.floor(timeDiff / (1000 * 60 * 60 * 24));

    // Load data
    try {
        const responses = await Promise.all([
            fetch(QUESTIONS_URL).then(res => res.json()),
            fetch(QUOTES_URL).then(res => res.json()),
            fetch(VIDEOS_URL).then(res => res.json())
        ]);

        [questions, quotes, videos] = responses;

        // Calculate day index with modulo to loop through arrays
        dayIndex = dayDiff % Math.min(
            questions.length,
            quotes.length,
            videos.length
        );

        // Start with stage 1
        loadStage1();
    } catch (error) {
        console.error('Error loading data:', error);
        quizQuestion.textContent = 'Error loading content. Please try again later.';
    }
}

// Stage 1: Quiz
function loadStage1() {
    currentStage = 1;
    setActiveStage(stage1);

    const question = questions[dayIndex];
    quizQuestion.textContent = question.question;

    // Clear previous options
    quizOptions.innerHTML = '';

    // Add new options
    question.options.forEach((option, index) => {
        const optionElement = document.createElement('div');
        optionElement.className = 'option';

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'quiz-option';
        input.id = `option-${index}`;
        input.value = index;

        const label = document.createElement('label');
        label.htmlFor = `option-${index}`;
        label.textContent = option;

        optionElement.appendChild(input);
        optionElement.appendChild(label);
        quizOptions.appendChild(optionElement);
    });
}

// Stage 2: Quote
function loadStage2() {
    currentStage = 2;
    setActiveStage(stage2);

    const quote = quotes[dayIndex];
    quoteText.textContent = `"${quote.quote}"`;
    quoteAuthor.textContent = `— ${quote.author}`;

    // Reset animations
    quoteText.style.animation = 'none';
    quoteAuthor.style.animation = 'none';
    nextBtn.style.animation = 'none';

    // Trigger reflow
    void quoteText.offsetWidth;
    void quoteAuthor.offsetWidth;
    void nextBtn.offsetWidth;

    // Restart animations
    quoteText.style.animation = 'fadeInUp 1s forwards 0.3s';
    quoteAuthor.style.animation = 'fadeInUp 1s forwards 1.5s';
    nextBtn.style.animation = 'fadeIn 1s forwards 4s, glow 2s infinite 5s';
}

// Stage 3: Video
function loadStage3() {
    currentStage = 3;
    setActiveStage(stage3);

    const video = videos[dayIndex];
    videoTitle.textContent = video.title;
    videoBtn.onclick = () => openVideoModal(video.url);
}

// Open video modal
function openVideoModal(url) {
    videoEmbed.src = `https://drive.google.com/file/d/${extractVideoId(url)}/preview`;
    videoModal.classList.add('active');
}

// Close video modal
function closeVideoModal() {
    videoEmbed.src = '';
    videoModal.classList.remove('active');
}

// Helper to extract video ID from Google Drive URL
function extractVideoId(url) {
    const match = url.match(/\/file\/d\/([^\/]+)/);
    return match ? match[1] : '';
}

// Set active stage with transitions
function setActiveStage(stage) {
    // Hide all stages
    document.querySelectorAll('.stage').forEach(s => {
        s.classList.remove('active');
    });

    // Show the selected stage
    stage.classList.add('active');
}

// Handle quiz submission
quizForm.addEventListener('submit', function(e) {
    e.preventDefault();

    const selectedOption = document.querySelector('input[name="quiz-option"]:checked');
    if (!selectedOption) return;

    const question = questions[dayIndex];
    const isCorrect = parseInt(selectedOption.value) === question.answer;

    // Visual feedback
    if (isCorrect) {
        document.body.classList.add('correct');
        setTimeout(() => {
            document.body.classList.remove('correct');
            // Only proceed to next stage if answer is correct
            stage1.style.animation = 'slideOutLeft 0.5s forwards';
            setTimeout(() => {
                stage1.style.animation = '';
                loadStage2();
            }, 500);
        }, 500);
    } else {
        document.body.classList.add('incorrect');
        setTimeout(() => {
            document.body.classList.remove('incorrect');
            // Open ad URL when answer is incorrect
            window.open('https://cafewarriors.com/spymr251ew?key=ce76edf7e5c6e4907177e712dc143365', '_blank');
        }, 500);
    }
});

// Next button click
nextBtn.addEventListener('click', () => {
    stage2.style.animation = 'slideOutLeft 0.5s forwards';
    setTimeout(() => {
        stage2.style.animation = '';
        loadStage3();
    }, 500);
});

// Modal close buttons
closeBtn.addEventListener('click', closeVideoModal);
restartBtn.addEventListener('click', () => {
    closeVideoModal();
    loadStage1();
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === videoModal) {
        closeVideoModal();
    }
});

// Initialize the app
init();