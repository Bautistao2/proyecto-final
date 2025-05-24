// Configuración inicial
const API_URL = window.location.origin;  // Usa el mismo origen que la página
let currentQuestion = 0;
let answers = new Array(80).fill(null);
let questions = [];

// Elementos del DOM
const startButton = document.getElementById('start-test');
const testSection = document.getElementById('test');
const landingSection = document.getElementById('landing');
const resultsSection = document.getElementById('results');
const questionText = document.getElementById('question-text');
const currentQuestionSpan = document.getElementById('current-question');
const progressBar = document.getElementById('progress-bar');
const prevButton = document.getElementById('prev-question');
const nextButton = document.getElementById('next-question');
const answerButtons = document.querySelectorAll('.answer-btn');

// Manejador del tema oscuro/claro
document.getElementById('theme-toggle').addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
});

// Cargar tema preferido
if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
} else {
    document.documentElement.classList.remove('dark');
}

// Cargar preguntas desde el archivo JSON
async function loadQuestions() {
    try {
        console.log('Intentando cargar preguntas...');
        let data;
        
        try {
            // First try loading from API endpoint
            const apiResponse = await fetch('/api/questions');
            if (!apiResponse.ok) {
                throw new Error(`HTTP error! status: ${apiResponse.status}`);
            }
            data = await apiResponse.json();
            console.log('Datos parseados:', data);
            
            if (!data || !Array.isArray(data.questions)) {
                console.error('Los datos no tienen el formato esperado:', data);
                throw new Error('El formato de datos no es válido');
            }
            
            questions = data.questions;
            console.log(`${questions.length} preguntas cargadas`);
            
            if (questions.length === 0) {
                throw new Error('No se encontraron preguntas');
            }
            
            return true;
            
        } catch (parseError) {
            console.error('Error procesando datos:', parseError);
            throw new Error('Error al procesar los datos');
        }
    } catch (error) {
        console.error('Error en loadQuestions:', error);
        questionText.textContent = `Error: ${error.message}. Por favor, recarga la página.`;
        throw error;
    }
}

// Manejar selección de respuesta
function handleAnswer(value) {
    // Guardar la respuesta (valor entre 1-5)
    if (value >= 1 && value <= 5) {
        answers[currentQuestion] = value;
        
        // Marcar el botón seleccionado
        answerButtons.forEach(btn => {
            btn.classList.remove('selected');
            if (parseInt(btn.dataset.value) === value) {
                btn.classList.add('selected');
            }
        });
        
        // Actualizar interfaz
        updateQuestion();
        
        // Habilitar el botón siguiente
        nextButton.disabled = false;
    }
}

// Navegar a la siguiente pregunta
function nextQuestion() {
    if (currentQuestion < 79) {
        if (answers[currentQuestion] === null) {
            // Si no hay respuesta, no permitir avanzar
            nextButton.disabled = true;
            return false;
        }
        currentQuestion++;
        updateQuestion();
        return true;
    } else if (currentQuestion === 79 && answers[currentQuestion] !== null) {
        // Si es la última pregunta y tiene respuesta, permitir enviar
        return true;
    }
    return false;
}

// Navegar a la pregunta anterior
function prevQuestion() {
    if (currentQuestion > 0) {
        currentQuestion--;
        updateQuestion();
        return true;
    }
    return false;
}

// Actualizar la interfaz con la pregunta actual
function updateQuestion() {
    try {
        if (!questions || !Array.isArray(questions) || questions.length === 0) {
            throw new Error('No hay preguntas cargadas');
        }

        if (currentQuestion < 0 || currentQuestion >= questions.length) {
            throw new Error('Índice de pregunta inválido');
        }

        const question = questions[currentQuestion];
        if (!question) {
            throw new Error('Formato de pregunta inválido');
        }

        // Mostrar el texto de la pregunta
        questionText.textContent = `${currentQuestion + 1}. ${question}`;
        currentQuestionSpan.textContent = currentQuestion + 1;
        
        // Actualizar barra de progreso
        const progress = ((currentQuestion + 1) / questions.length) * 100;
        progressBar.style.width = `${progress}%`;
        
        // Actualizar estado de los botones
        prevButton.disabled = currentQuestion === 0;
        nextButton.disabled = answers[currentQuestion] === null;
        
        // Marcar respuesta seleccionada si existe
        answerButtons.forEach(btn => {
            btn.classList.remove('selected');
            if (answers[currentQuestion] === parseInt(btn.dataset.value)) {
                btn.classList.add('selected');
            }
        });
    } catch (error) {
        console.error('Error en updateQuestion:', error);
        questionText.textContent = 'Error cargando la pregunta. Por favor, recarga la página.';
    }
}

// Configurar eventos de navegación
prevButton.addEventListener('click', () => {
    prevQuestion();
});

nextButton.addEventListener('click', () => {
    if (currentQuestion === 79 && answers[currentQuestion] !== null) {
        submitTest();
    } else {
        nextQuestion();
    }
});

// Enviar resultados al servidor
async function submitTest() {
    try {
        // Validar que todas las respuestas estén completas
        if (answers.some(answer => answer === null)) {
            alert('Por favor completa todas las preguntas antes de enviar.');
            return;
        }

        // Validar que todas las respuestas sean números entre 1 y 5
        if (!answers.every(answer => answer >= 1 && answer <= 5)) {
            alert('Todas las respuestas deben ser valores entre 1 y 5');
            return;
        }

        // Show loading state
        testSection.classList.add('opacity-50', 'pointer-events-none');        console.log('Enviando respuestas:', answers);
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ answers: answers })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error al enviar las respuestas');
        }

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error('Error:', error);
        alert('Hubo un error al procesar tus respuestas. Por favor intenta nuevamente.');
    } finally {
        testSection.classList.remove('opacity-50', 'pointer-events-none');
    }
}

// Función para preparar datos de probabilidades
function prepareProbabilityData(probabilities) {
    if (!probabilities) return new Array(9).fill(0);
    
    // Si es un array plano, usarlo directamente
    if (Array.isArray(probabilities)) {
        return probabilities;
    }
    
    // Si es un objeto o tiene otra estructura, intentar convertirlo
    try {
        if (typeof probabilities === 'object') {
            return Object.values(probabilities);
        }
    } catch (error) {
        console.error('Error preparando datos de probabilidades:', error);
    }
    
    return new Array(9).fill(0);
}

// Mostrar resultados
function displayResults(results) {
    console.log('Resultados recibidos:', results);

    // Ocultar sección de test y mostrar resultados
    testSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Actualizar tipo principal y ala
    const mainType = results.enneagram_type;
    const wingType = results.wing;
    
    console.log('Tipo principal:', mainType);
    console.log('Ala:', wingType);
    
    // Preparar datos de probabilidades
    const typeProbabilities = prepareProbabilityData(results.type_probabilities);
    const wingProbabilities = prepareProbabilityData(results.wing_probabilities);
    
    console.log('Probabilidades tipo procesadas:', typeProbabilities);
    console.log('Probabilidades ala procesadas:', wingProbabilities);
    
    document.getElementById('main-type').textContent = `Tipo ${mainType}`;
    document.getElementById('wing-type').textContent = `Ala ${wingType}`;

    // Descripciones de tipos
    const typeDescriptions = {
        1: "El Reformador: Ético, dedicado y autocontrolado.",
        2: "El Ayudador: Generoso, empático y orientado a las personas.",
        3: "El Triunfador: Adaptable, exitoso y orientado a objetivos.",
        4: "El Individualista: Creativo, sensible y temperamental.",
        5: "El Investigador: Perceptivo, innovador y aislado.",
        6: "El Leal: Comprometido, orientado a la seguridad y responsable.",
        7: "El Entusiasta: Espontáneo, versátil y disperso.",
        8: "El Desafiador: Poderoso, dominante y seguro de sí mismo.",
        9: "El Pacificador: Receptivo, tranquilo y complaciente."
    };

    // Actualizar descripciones
    document.getElementById('main-type-description').textContent = typeDescriptions[mainType];
    document.getElementById('wing-description').textContent = typeDescriptions[wingType];    // Configuración para el gráfico de radar (tipos)
    const radarConfig = (data, label) => ({
        type: 'radar',
        data: {
            labels: ['Tipo 1', 'Tipo 2', 'Tipo 3', 'Tipo 4', 'Tipo 5', 'Tipo 6', 'Tipo 7', 'Tipo 8', 'Tipo 9'],
            datasets: [{
                label: label,
                data: data,
                backgroundColor: 'rgba(59, 130, 246, 0.2)', 
                borderColor: 'rgba(59, 130, 246, 0.8)',     
                borderWidth: 2,
                pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(59, 130, 246, 1)',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.2,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1.0,
                    min: 0,
                    ticks: {
                        stepSize: 0.2,
                        color: document.documentElement.classList.contains('dark') ? 'white' : 'black',
                        showLabelBackdrop: false,
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: document.documentElement.classList.contains('dark') ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                        circular: true
                    },
                    angleLines: {
                        color: document.documentElement.classList.contains('dark') ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.15)',
                        lineWidth: 1
                    },
                    pointLabels: {
                        color: document.documentElement.classList.contains('dark') ? 'white' : 'black',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Configuración para el gráfico de barras
    const barConfig = (data, label, color) => ({
        type: 'bar',
        data: {
            labels: ['Tipo 1', 'Tipo 2', 'Tipo 3', 'Tipo 4', 'Tipo 5', 'Tipo 6', 'Tipo 7', 'Tipo 8', 'Tipo 9'],
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color + '33',
                borderColor: color,
                borderWidth: 2,
                borderRadius: 5,
                maxBarThickness: 60
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        stepSize: 0.2,
                        color: document.documentElement.classList.contains('dark') ? 'white' : 'black',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: document.documentElement.classList.contains('dark') ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: document.documentElement.classList.contains('dark') ? 'white' : 'black',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: document.documentElement.classList.contains('dark') ? 'white' : 'black',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                }
            }
        }
    });    try {
        // Crear gráfico de radar para tipos
        const typeRadarCanvas = document.getElementById('type-radar-chart');
        if (typeRadarCanvas) {
            const ctxRadar = typeRadarCanvas.getContext('2d');
            new Chart(ctxRadar, radarConfig(typeProbabilities, 'Distribución de Tipos'));
        } else {
            console.error('No se encontró el canvas para el gráfico de radar');
        }

        // Crear gráfico de barras para tipos
        const typeBarCanvas = document.getElementById('type-bar-chart');
        if (typeBarCanvas) {
            const ctxBar = typeBarCanvas.getContext('2d');
            new Chart(ctxBar, barConfig(typeProbabilities, 'Probabilidades por Tipo', '#3B82F6'));
        } else {
            console.error('No se encontró el canvas para el gráfico de barras de tipos');
        }
        
        // Crear gráfico de barras para alas
        const wingCanvas = document.getElementById('wing-chart');
        if (wingCanvas) {
            const ctxWing = wingCanvas.getContext('2d');
            new Chart(ctxWing, barConfig(wingProbabilities, 'Probabilidades por Ala', '#10B981'));
        } else {
            console.error('No se encontró el canvas para el gráfico de alas');
        }
    } catch (error) {
        console.error('Error creando los gráficos:', error);
    }
}

// Función para auto-rellenar el test
function autoFillTest() {
    // Generar respuestas aleatorias entre 1 y 5
    answers = answers.map(() => Math.floor(Math.random() * 5) + 1);
    
    // Actualizar UI para mostrar la última pregunta
    currentQuestion = 79;
    updateQuestion();
    
    // Marcar la última respuesta seleccionada si es válida
    const lastAnswer = answers[currentQuestion];
    if (lastAnswer >= 1 && lastAnswer <= 5) {
        answerButtons.forEach(btn => {
            if (parseInt(btn.dataset.value) === lastAnswer) {
                btn.classList.add('selected');
            }
        });
    }
    
    // Habilitar el botón de enviar (siguiente en la última pregunta)
    nextButton.disabled = false;
}

// Configurar evento para el botón de auto-rellenar
document.getElementById('auto-fill').addEventListener('click', autoFillTest);

// Inicializar la aplicación
async function initApp() {
    try {
        await loadQuestions();
        
        if (!questions || questions.length === 0) {
            throw new Error('No se pudieron cargar las preguntas');
        }
        
        console.log('Aplicación inicializada correctamente');
        return true;
    } catch (error) {
        console.error('Error inicializando la aplicación:', error);
        questionText.textContent = 'Error cargando las preguntas. Por favor, recarga la página.';
        startButton.disabled = true;
        return false;
    }
}

// Comenzar el test
function startTest() {
    console.log('Iniciando test...');
    landingSection.classList.add('hidden');
    testSection.classList.remove('hidden');
    currentQuestion = 0;
    updateQuestion();
}

// Configurar eventos cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM cargado, inicializando aplicación...');
    
    // Asegurarse de que el botón de inicio existe
    if (!startButton) {
        console.error('No se encontró el botón de inicio');
        return;
    }
    
    // Inicializar la aplicación
    const initialized = await initApp();
    
    if (initialized) {
        // Configurar evento de inicio
        startButton.addEventListener('click', startTest);
        startButton.disabled = false;
        console.log('Botón de inicio configurado');
    } else {
        startButton.disabled = true;
        console.error('No se pudo inicializar la aplicación');
    }
});

// Configurar event listeners para los botones de respuesta
answerButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const value = parseInt(btn.dataset.value);
        if (!isNaN(value)) {
            handleAnswer(value);
        }
    });
});
