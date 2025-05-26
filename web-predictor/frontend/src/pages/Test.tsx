import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Heading,
  Text,
  VStack,
  HStack,
  Progress,
  Radio,
  RadioGroup,
  Flex,
  Card,
  CardBody,
  Spinner,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react';
import { apiService, Question } from '../services/api';

const likertOptions = [
  { value: 1, label: 'Totalmente en desacuerdo' },
  { value: 2, label: 'En desacuerdo' },
  { value: 3, label: 'Neutral' },
  { value: 4, label: 'De acuerdo' },
  { value: 5, label: 'Totalmente de acuerdo' }
];

const Test: React.FC = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const navigate = useNavigate();
  const toast = useToast();
  
  const questionsPerPage = 10;
  const totalPages = Math.ceil(questions.length / questionsPerPage);
  const progress = (Object.keys(answers).length / questions.length) * 100;
  
  // Fetch questions on component mount
  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        setLoading(true);
        const fetchedQuestions = await apiService.getQuestions();
        setQuestions(fetchedQuestions);
        setError(null);
      } catch (err) {
        console.error('Error fetching questions:', err);
        setError('No se pudieron cargar las preguntas. Por favor, intenta de nuevo más tarde.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchQuestions();
  }, []);
  
  // Get current page questions
  const getCurrentPageQuestions = useCallback(() => {
    const startIndex = (currentPage - 1) * questionsPerPage;
    return questions.slice(startIndex, startIndex + questionsPerPage);
  }, [questions, currentPage, questionsPerPage]);
  
  // Handle answer selection
  const handleAnswerChange = (questionId: number, value: number) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };
  
  // Handle page navigation
  const goToNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
      window.scrollTo(0, 0);
    }
  };
  
  const goToPrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
      window.scrollTo(0, 0);
    }
  };
  
  // Submit answers
  const handleSubmit = async () => {
    // Check if all questions have been answered
    if (Object.keys(answers).length !== questions.length) {
      toast({
        title: 'Respuestas incompletas',
        description: 'Por favor responde todas las preguntas antes de enviar.',
        status: 'warning',
        duration: 5000,
        isClosable: true
      });
      return;
    }
    
    try {
      setSubmitting(true);
      
      // Convert answers object to array in the correct order
      const answersArray = questions.map(q => answers[q.id]);
      
      // Submit answers to API
      const result = await apiService.submitAnswers(answersArray);
      
      // Store results in sessionStorage and navigate to results page
      sessionStorage.setItem('enneagramResults', JSON.stringify(result));
      navigate('/results');
      
    } catch (err) {
      console.error('Error submitting answers:', err);
      toast({
        title: 'Error al enviar respuestas',
        description: 'Hubo un problema al procesar tus respuestas. Por favor, intenta de nuevo.',
        status: 'error',
        duration: 5000,
        isClosable: true
      });
    } finally {
      setSubmitting(false);
    }
  };
  
  const fillCurrentBlockRandomly = () => {
    const currentQuestions = getCurrentPageQuestions();
    setAnswers(prev => {
      const updated = { ...prev };
      currentQuestions.forEach(q => {
        updated[q.id] = Math.floor(Math.random() * 5) + 1;
      });
      return updated;
    });
  };
  
  if (loading) {
    return (
      <Box textAlign="center" py={10}>
        <Spinner size="xl" color="primary.500" />
        <Text mt={4}>Cargando preguntas...</Text>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Alert
        status="error"
        variant="subtle"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        textAlign="center"
        height="200px"
        borderRadius="md"
        my={8}
      >
        <AlertIcon boxSize="40px" mr={0} />
        <AlertTitle mt={4} mb={1} fontSize="lg">
          Error al cargar las preguntas
        </AlertTitle>
        <AlertDescription maxWidth="sm">
          {error}
        </AlertDescription>
      </Alert>
    );
  }
  
  const currentQuestions = getCurrentPageQuestions();
  
  return (
    <Box>
      <Heading mb={4} color="primary.600">Test de Eneagrama</Heading>
      <Text mb={8}>Responde todas las preguntas para descubrir tu tipo de eneagrama.</Text>
      
      {/* Progress bar */}
      <Box mb={8}>
        <Flex justify="space-between" mb={2}>
          <Text fontSize="sm">{Object.keys(answers).length} de {questions.length} respondidas</Text>
          <Text fontSize="sm">{Math.round(progress)}%</Text>
        </Flex>
        <Progress value={progress} colorScheme="blue" borderRadius="md" />
      </Box>
      
      {/* Questions */}
      <VStack spacing={6} align="stretch" mb={8}>
        {/* Botón para rellenar aleatoriamente el bloque actual */}
        <Button mb={2} colorScheme="orange" variant="outline" size="sm" onClick={fillCurrentBlockRandomly}>
          Rellenar bloque aleatoriamente
        </Button>
        {currentQuestions.map((question) => (
          <Card key={question.id} variant="outline">
            <CardBody>
              <Box mb={4}>
                <Text fontWeight="medium">
                  {question.id}. {question.text}
                </Text>
                {/* Categoría eliminada */}
              </Box>
              
              <RadioGroup 
                onChange={(val) => handleAnswerChange(question.id, parseInt(val))} 
                value={answers[question.id]?.toString() || undefined}
              >
                <HStack spacing={4} wrap="wrap">
                  {likertOptions.map(option => (
                    <Radio key={option.value} value={option.value.toString()}>
                      {option.label}
                    </Radio>
                  ))}
                </HStack>
              </RadioGroup>
            </CardBody>
          </Card>
        ))}
      </VStack>
      
      {/* Navigation */}
      <Flex justify="space-between" align="center">
        <Button 
          onClick={goToPrevPage} 
          isDisabled={currentPage === 1}
          colorScheme="gray"
        >
          Anterior
        </Button>
        
        <Text>
          Página {currentPage} de {totalPages}
        </Text>
        
        {currentPage < totalPages ? (
          <Button 
            onClick={goToNextPage} 
            colorScheme="blue"
            isDisabled={currentQuestions.some(q => !answers[q.id])} // Deshabilita si falta alguna respuesta
          >
            Siguiente
          </Button>
        ) : (
          <Button 
            onClick={handleSubmit} 
            colorScheme="green" 
            isLoading={submitting}
            loadingText="Enviando"
            isDisabled={currentQuestions.some(q => !answers[q.id])} // Deshabilita si falta alguna respuesta
          >
            Finalizar y Ver Resultados
          </Button>
        )}
      </Flex>
    </Box>
  );
};

export default Test;
