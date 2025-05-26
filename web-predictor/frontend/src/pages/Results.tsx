import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Heading,
  Text,
  VStack,
  HStack,
  Button,
  Flex,
  Card,
  CardBody,
  CardHeader,
  Divider,
  CircularProgress,
  CircularProgressLabel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  SimpleGrid,
  useColorModeValue
} from '@chakra-ui/react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface PredictionResponse {
  enneagram_type: number;
  wing: number;
  type_probabilities: number[];
  wing_probabilities: number[];
}

const enneagramTypes = [
  { number: 1, name: "El Reformador", emoji: "🛠️", description: "¡El perfeccionista del grupo! Siempre buscando mejorar todo, hasta el café de la mañana. Si algo puede hacerse mejor, ahí estás tú, con tu manual de instrucciones y tu regla. A veces te cuesta relajarte, pero tu ética inspira a todos." },
  { number: 2, name: "El Ayudador", emoji: "🤝", description: "El alma generosa y el hombro donde todos lloran. Te encanta ayudar, a veces tanto que te olvidas de ti mismo. Si hay un cumpleaños, tú llevas la torta y los abrazos. Solo recuerda: también mereces recibir cariño." },
  { number: 3, name: "El Triunfador", emoji: "🏆", description: "El campeón de la productividad. Siempre con la agenda llena y la sonrisa lista para la foto. Te gusta brillar y que te reconozcan, pero no olvides que eres valioso incluso sin medallas. ¡Eres el Messi de tu grupo!" },
  { number: 4, name: "El Individualista", emoji: "🎨", description: "El artista dramático y profundo. Nadie siente como tú, y tu playlist lo sabe. Buscas ser único y auténtico, aunque a veces te pierdes en tus emociones. ¡El mundo sería muy aburrido sin tu toque especial!" },
  { number: 5, name: "El Investigador", emoji: "🔬", description: "El Sherlock Holmes del eneagrama. Curioso, analítico y siempre con un dato interesante bajo la manga. Te encanta tu espacio y tu tiempo a solas, pero recuerda: compartir tus ideas puede ser tan genial como descubrirlas." },
  { number: 6, name: "El Leal", emoji: "🛡️", description: "El guardián del grupo. Siempre preparado para cualquier emergencia (¡y con un botiquín en la mochila!). Valoras la seguridad y la confianza, aunque a veces la duda te visita. Tus amigos saben que pueden contar contigo pase lo que pase." },
  { number: 7, name: "El Entusiasta", emoji: "🎉", description: "El alma de la fiesta y el planificador de aventuras. Siempre tienes una idea divertida y mil proyectos en marcha. Te cuesta aburrirte, pero también parar. ¡Contigo la vida es una montaña rusa de alegría!" },
  { number: 8, name: "El Desafiador", emoji: "⚡", description: "El líder nato, fuerte y directo. No le temes a los desafíos y siempre dices lo que piensas. A veces puedes parecer intenso, pero tu energía mueve montañas (¡y grupos de WhatsApp!)." },
  { number: 9, name: "El Pacificador", emoji: "🕊️", description: "El zen del grupo. Tranquilo, conciliador y siempre buscando la armonía. Te cuesta el conflicto, pero tu paz es contagiosa. Si todos fueran como tú, ¡el mundo sería un spa!" }
];

const getEnneagramInfo = (type: number) => {
  return enneagramTypes.find(t => t.number === type) || enneagramTypes[0];
};

// Descripciones ampliadas por aspecto para cada tipo
const enneagramTypeAspects: Record<number, {
  love: string;
  friendship: string;
  work: string;
}> = {
  1: {
    love: "En el amor, eres leal y buscas relaciones auténticas, aunque a veces puedes ser un poco exigente con tu pareja (¡pero solo porque quieres lo mejor!).",
    friendship: "Como amigo, eres el consejero del grupo, siempre dispuesto a dar una mano y corregir la ortografía en los memes.",
    work: "En el trabajo, eres el que revisa dos veces los informes y mantiene la oficina en orden. ¡Eres el Excel personificado!"
  },
  2: {
    love: "En el amor, eres detallista y cariñoso, el primero en recordar aniversarios y preparar sorpresas. ¡Cuidado con darlo todo y olvidarte de ti!",
    friendship: "Como amigo, eres el que organiza las juntadas y manda mensajes de buenos días. Si alguien necesita ayuda, ahí estás tú, con galletitas y consejos.",
    work: "En el trabajo, eres el alma del equipo, siempre dispuesto a colaborar y a traer café. A veces te cuesta decir que no, pero todos te adoran." 
  },
  3: {
    love: "En el amor, eres seductor y motivador, siempre buscando crecer en pareja. Te gusta que te admiren, pero también sabes inspirar a tu media naranja.",
    friendship: "Como amigo, eres el que impulsa a todos a cumplir sus metas y a salir bien en las fotos. ¡Nunca faltan tus historias de éxito en las reuniones!",
    work: "En el trabajo, eres el crack: competitivo, eficiente y siempre buscando el próximo logro. ¡Cuidado con el burnout!"
  },
  4: {
    love: "En el amor, eres intenso y romántico, capaz de escribir poemas o playlists para tu pareja. A veces te pierdes en tus emociones, pero tu autenticidad enamora.",
    friendship: "Como amigo, eres el confidente profundo, el que escucha y comprende sin juzgar. Tus regalos siempre son únicos y con significado.",
    work: "En el trabajo, aportas creatividad y sensibilidad. Ves lo que otros no ven, aunque a veces te distraes soñando despierto." 
  },
  5: {
    love: "En el amor, eres reservado pero leal. Prefieres una charla profunda a una salida ruidosa. Tu pareja sabe que contigo, el silencio también es compañía.",
    friendship: "Como amigo, eres el que tiene datos curiosos para cada ocasión y ayuda a resolver problemas con lógica. A veces te pierdes en tus pensamientos, pero siempre vuelves con una solución ingeniosa.",
    work: "En el trabajo, eres el analista estrella. Te encanta investigar y encontrar la mejor manera de hacer las cosas. ¡Eres el Google humano del equipo!"
  },
  6: {
    love: "En el amor, eres protector y fiel. Te preocupas por el bienestar de tu pareja y siempre tienes un plan B para las citas.",
    friendship: "Como amigo, eres el que nunca olvida un cumpleaños y siempre está para escuchar. Si hay un problema, tú ya tienes la solución (¡y el seguro de viaje!).",
    work: "En el trabajo, eres el que mantiene todo bajo control. Eres confiable y responsable, aunque a veces la ansiedad te juega una mala pasada."
  },
  7: {
    love: "En el amor, eres divertido y espontáneo. Siempre tienes un plan sorpresa y haces que cada día sea una aventura.",
    friendship: "Como amigo, eres el alma de la fiesta y el que propone los viajes. ¡Contigo nunca hay aburrimiento!",
    work: "En el trabajo, eres creativo y entusiasta. Te encantan los nuevos proyectos, aunque a veces te cuesta terminarlos todos." 
  },
  8: {
    love: "En el amor, eres apasionado y protector. Dices lo que piensas y defiendes a los tuyos con uñas y dientes.",
    friendship: "Como amigo, eres el líder del grupo, el que organiza y pone límites. A veces puedes ser intenso, pero tu lealtad es inquebrantable.",
    work: "En el trabajo, eres el que toma decisiones y enfrenta los desafíos. No le temes a nada, pero recuerda escuchar a los demás." 
  },
  9: {
    love: "En el amor, eres comprensivo y paciente. Evitas discusiones y buscas la armonía en la pareja.",
    friendship: "Como amigo, eres el mediador, el que calma las aguas y escucha a todos. Tu paz es contagiosa.",
    work: "En el trabajo, eres el que une al equipo y evita conflictos. Prefieres la cooperación a la competencia. ¡Eres el zen de la oficina!"
  }
};

// Descripciones ampliadas por aspecto para cada ala (puedes personalizar más si lo deseas)
const enneagramWingAspects: Record<number, {
  love: string;
  friendship: string;
  work: string;
}> = {
  1: {
    love: "Como ala, aporta un toque de perfeccionismo y ética a tu forma de amar.",
    friendship: "En la amistad, suma responsabilidad y consejos útiles (¡a veces demasiados!).",
    work: "En el trabajo, te ayuda a ser más ordenado y a buscar la excelencia en todo lo que haces."
  },
  2: {
    love: "Como ala, añade calidez y generosidad a tus relaciones amorosas.",
    friendship: "En la amistad, te hace más atento y dispuesto a ayudar.",
    work: "En el trabajo, potencia tu espíritu colaborativo y tu empatía con los compañeros."
  },
  3: {
    love: "Como ala, suma ambición y ganas de destacar en el amor (¡y en las citas!).",
    friendship: "En la amistad, te vuelve más motivador y competitivo (en los juegos de mesa, seguro).",
    work: "En el trabajo, te impulsa a buscar logros y reconocimiento extra."
  },
  4: {
    love: "Como ala, aporta sensibilidad y creatividad a tu vida amorosa.",
    friendship: "En la amistad, te hace más empático y original.",
    work: "En el trabajo, suma un toque artístico y diferente a todo lo que haces."
  },
  5: {
    love: "Como ala, añade análisis y reflexión a tus relaciones.",
    friendship: "En la amistad, te vuelve más observador y buen consejero.",
    work: "En el trabajo, potencia tu capacidad de investigación y tu lógica."
  },
  6: {
    love: "Como ala, suma lealtad y protección a tu forma de amar.",
    friendship: "En la amistad, te hace más confiable y siempre presente.",
    work: "En el trabajo, refuerza tu sentido de responsabilidad y previsión."
  },
  7: {
    love: "Como ala, aporta alegría y espontaneidad a tus relaciones.",
    friendship: "En la amistad, te vuelve más divertido y siempre listo para un plan.",
    work: "En el trabajo, suma creatividad y energía positiva a los proyectos."
  },
  8: {
    love: "Como ala, añade fuerza y decisión a tu vida amorosa.",
    friendship: "En la amistad, te hace más protector y directo.",
    work: "En el trabajo, potencia tu liderazgo y tu capacidad de acción."
  },
  9: {
    love: "Como ala, suma calma y paciencia a tus relaciones.",
    friendship: "En la amistad, te vuelve más conciliador y relajado.",
    work: "En el trabajo, ayuda a mantener la paz y el buen ambiente en el equipo."
  }
};

const Results: React.FC = () => {
  // Hooks deben ir siempre al inicio
  const cardBg = useColorModeValue('white', 'gray.700');
  const cardBgHighlight = useColorModeValue('blue.50', 'blue.900');
  const radarMainColor = useColorModeValue('rgba(49,130,206,1)', 'rgba(144,205,244,1)');
  const radarMainBg = useColorModeValue('rgba(49,130,206,0.2)', 'rgba(144,205,244,0.2)');
  const radarWingColor = useColorModeValue('rgba(56,178,172,1)', 'rgba(81,230,220,1)');
  const radarWingBg = useColorModeValue('rgba(56,178,172,0.15)', 'rgba(81,230,220,0.15)');
  const radarGridColor = useColorModeValue('#CBD5E0', '#4A5568');
  const radarLabelColor = useColorModeValue('#2D3748', '#E2E8F0');

  const [results, setResults] = useState<PredictionResponse | null>(null);
  const navigate = useNavigate();
  
  useEffect(() => {
    // Get results from sessionStorage
    const storedResults = sessionStorage.getItem('enneagramResults');
    
    if (storedResults) {
      try {
        const parsedResults = JSON.parse(storedResults);
        setResults(parsedResults);
      } catch (err) {
        console.error('Error parsing results:', err);
      }
    }
  }, []);
  
  const handleRetakeTest = () => {
    // Clear stored results
    sessionStorage.removeItem('enneagramResults');
    navigate('/test');
  };
  
  if (!results) {
    return (
      <Alert
        status="info"
        variant="subtle"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        textAlign="center"
        height="200px"
        borderRadius="md"
      >
        <AlertIcon boxSize="40px" mr={0} />
        <AlertTitle mt={4} mb={1} fontSize="lg">
          No hay resultados disponibles
        </AlertTitle>
        <AlertDescription maxWidth="sm">
          No se han encontrado resultados del test. Por favor, realiza el test para ver tus resultados.
          <Button 
            mt={4}
            colorScheme="blue"
            onClick={() => navigate('/test')}
          >
            Ir al Test
          </Button>
        </AlertDescription>
      </Alert>
    );
  }
  
  const mainType = getEnneagramInfo(results.enneagram_type);
  const wingType = getEnneagramInfo(results.wing);
  
  // Calculate percentage for main type and wing
  const mainTypePercentage = Math.round(results.type_probabilities[results.enneagram_type - 1] * 100);
  const wingTypePercentage = Math.round(results.wing_probabilities[results.wing - 1] * 100);
  
  // Radar chart data for types
  const radarData = {
    labels: enneagramTypes.map(t => `${t.emoji} Tipo ${t.number}\n${t.name}`),
    datasets: [
      {
        label: 'Probabilidad Tipo',
        data: results.type_probabilities.map(p => Math.round(p * 100)),
        backgroundColor: radarMainBg,
        borderColor: radarMainColor,
        pointBackgroundColor: radarMainColor,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: radarMainColor,
      },
      {
        label: 'Probabilidad Ala',
        data: results.wing_probabilities.map(p => Math.round(p * 100)),
        backgroundColor: radarWingBg,
        borderColor: radarWingColor,
        pointBackgroundColor: radarWingColor,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: radarWingColor,
      }
    ]
  };
  const radarOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: { color: radarLabelColor }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => `${context.dataset.label}: ${context.formattedValue}%`
        }
      }
    },
    scales: {
      r: {
        angleLines: { color: radarGridColor },
        grid: { color: radarGridColor },
        pointLabels: { color: radarLabelColor, font: { size: 14 } },
        ticks: {
          color: radarLabelColor,
          stepSize: 20,
          callback: (v: any) => `${v}%`
        },
        min: 0,
        max: 100
      }
    }
  };
  
  return (
    <Box>
      <Heading mb={2} color="primary.600">Resultados del Test</Heading>
      <Text mb={8} fontSize="lg">
        Basado en tus respuestas, aquí está tu perfil de eneagrama:
      </Text>
      
      {/* Main Results */}
      <Card bg={cardBgHighlight} mb={8}>
        <CardBody>
          <Flex direction={{ base: 'column', md: 'row' }} align="center" justify="space-between">
            <VStack align={{ base: 'center', md: 'start' }} spacing={3} flex={1}>
              <Heading size="lg">
                Tipo {mainType.number}: {mainType.name}
              </Heading>
              <Heading size="md" color="gray.600">
                con Ala {wingType.number}
              </Heading>
              <Text fontSize="lg">{mainType.description}</Text>
            </VStack>
            
            <HStack spacing={6} mt={{ base: 6, md: 0 }}>
              <CircularProgress 
                value={mainTypePercentage} 
                size="120px" 
                color="primary.500" 
                thickness="8px"
              >
                <CircularProgressLabel>
                  <VStack spacing={0}>
                    <Text fontWeight="bold" fontSize="xl">{mainTypePercentage}%</Text>
                    <Text fontSize="sm">Tipo {mainType.number}</Text>
                  </VStack>
                </CircularProgressLabel>
              </CircularProgress>
              
              <CircularProgress 
                value={wingTypePercentage} 
                size="120px" 
                color="teal.400" 
                thickness="8px"
              >
                <CircularProgressLabel>
                  <VStack spacing={0}>
                    <Text fontWeight="bold" fontSize="xl">{wingTypePercentage}%</Text>
                    <Text fontSize="sm">Ala {wingType.number}</Text>
                  </VStack>
                </CircularProgressLabel>
              </CircularProgress>
            </HStack>
          </Flex>
        </CardBody>
      </Card>
      
      {/* Detailed Results - Radar Chart */}
      <Heading size="md" mb={4}>Probabilidades Detalladas</Heading>
      <Card bg={cardBg} mb={8}>
        <CardBody>
          <Box w={{ base: '100%', md: '70%' }} mx="auto">
            <Radar data={radarData} options={radarOptions} style={{ maxHeight: 400 }} />
          </Box>
        </CardBody>
      </Card>
      
      {/* Understanding Your Type */}
      <Card bg={cardBg} mb={8}>
        <CardHeader>
          <Heading size="md">Comprendiendo tu tipo {mainType.number}w{wingType.number}</Heading>
        </CardHeader>
        <CardBody>
          <Text mb={4}>
            Como un tipo {mainType.number} con ala {wingType.number}, tu personalidad combina las características de ambos tipos:
          </Text>
          <VStack align="stretch" spacing={4}>
            <Box>
              <Heading size="sm" mb={2} color="primary.600">Tipo Principal ({mainType.number} {mainType.emoji} - {mainType.name})</Heading>
              <Text mb={2}>{mainType.description}</Text>
              <Text><b>En el amor:</b> {enneagramTypeAspects[mainType.number].love}</Text>
              <Text><b>En la amistad:</b> {enneagramTypeAspects[mainType.number].friendship}</Text>
              <Text><b>En el trabajo:</b> {enneagramTypeAspects[mainType.number].work}</Text>
            </Box>
            <Divider />
            <Box>
              <Heading size="sm" mb={2} color="teal.600">Influencia del Ala ({wingType.number} {wingType.emoji} - {wingType.name})</Heading>
              <Text mb={2}>{wingType.description}</Text>
              <Text><b>En el amor:</b> {enneagramWingAspects[wingType.number].love}</Text>
              <Text><b>En la amistad:</b> {enneagramWingAspects[wingType.number].friendship}</Text>
              <Text><b>En el trabajo:</b> {enneagramWingAspects[wingType.number].work}</Text>
            </Box>
          </VStack>
        </CardBody>
      </Card>
      
      {/* Action Buttons */}
      <Flex justify="center" gap={4}>
        <Button 
          colorScheme="blue" 
          onClick={handleRetakeTest}
        >
          Realizar el Test de Nuevo
        </Button>
        <Button 
          as="a" 
          href="/" 
          colorScheme="gray"
        >
          Volver al Inicio
        </Button>
      </Flex>
    </Box>
  );
};

export default Results;
