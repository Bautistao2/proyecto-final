import React from 'react';
import { 
  Box, 
  Heading, 
  Text, 
  Button, 
  VStack,
  HStack,
  Image,
  Container,
  SimpleGrid,
  Icon,
  Stack,
  useColorModeValue
} from '@chakra-ui/react';
import { FaChartPie, FaLightbulb, FaUser } from 'react-icons/fa6';
import { IconType } from 'react-icons';
import { ElementType } from 'react';
import { Link as RouterLink } from 'react-router-dom';

interface FeatureProps {
  title: string;
  text: string;
  icon: IconType;
}

const Feature = ({ title, text, icon }: FeatureProps) => {
  return (
    <Stack align={'center'} textAlign={'center'}>
      <Flex
        w={16}
        h={16}
        alignItems={'center'}
        justifyContent={'center'}
        color={'white'}
        rounded={'full'}
        bg={'primary.500'}
        mb={1}
      >
        {/* Renderiza el icono como JSX directamente */}
        {icon && icon({ size: 40 })}
      </Flex>
      <Text fontWeight={600}>{title}</Text>
      <Text color={'gray.600'}>{text}</Text>
    </Stack>
  );
};

const Landing: React.FC = () => {
  return (
    <Box>
      {/* Hero Section */}
      <Container maxW={'7xl'}>
        <Stack
          align={'center'}
          spacing={{ base: 8, md: 10 }}
          py={{ base: 20, md: 28 }}
          direction={{ base: 'column', md: 'row' }}>
          <Stack flex={1} spacing={{ base: 5, md: 10 }}>
            <Heading
              lineHeight={1.1}
              fontWeight={600}
              fontSize={{ base: '3xl', sm: '4xl', lg: '6xl' }}>
              <Text
                as={'span'}
                position={'relative'}
                color={'primary.500'}>
                Descubre tu Eneagrama
              </Text>
              <br />
              <Text as={'span'} color={'gray.700'}>
                Conoce tu personalidad
              </Text>
            </Heading>
            <Text color={'gray.500'}>
              Nuestro sistema de predicción de eneagrama utiliza algoritmos avanzados de inteligencia artificial para determinar tu tipo de eneagrama y ala con alta precisión. Responde 80 preguntas sencillas y obtén resultados inmediatos.
            </Text>
            <Stack
              spacing={{ base: 4, sm: 6 }}
              direction={{ base: 'column', sm: 'row' }}>
              <Button
                as={RouterLink}
                to='/test'
                rounded={'full'}
                size={'lg'}
                fontWeight={'normal'}
                px={6}
                colorScheme={'blue'}
                bg={'primary.500'}
                _hover={{ bg: 'primary.600' }}>
                Comenzar Test
              </Button>
              <Button
                rounded={'full'}
                size={'lg'}
                fontWeight={'normal'}
                px={6}
                leftIcon={<span>{FaLightbulb && FaLightbulb({ size: 20 })}</span>}>
                Aprender más
              </Button>
            </Stack>
          </Stack>
          <Flex
            flex={1}
            justifyContent={'center'}
            alignItems={'center'}
            position={'relative'}
            w={'full'}>
            <Box
              position={'relative'}
              height={{ base: '250px', md: '400px' }}
              minH={{ base: '250px', md: '400px' }}
              rounded={'2xl'}
              boxShadow={'2xl'}
              width={'full'}
              overflow={'hidden'}>
              <Image
                alt={'Eneagrama'}
                fit={'cover'}
                align={'center'}
                w={'100%'}
                h={'100%'}
                src={'/imagen_principal.png'}
              />
            </Box>
          </Flex>
        </Stack>
      </Container>

      {/* Feature Section */}
      <Box py={12} bg={useColorModeValue('gray.50', 'gray.700')}>
        <Container maxW={'7xl'}>
          <VStack spacing={2} textAlign={'center'} mb={12}>
            <Heading as={'h2'} fontSize={'3xl'} color={'primary.500'}>
              ¿Por qué utilizar nuestro test?
            </Heading>
            <Text color={'gray.500'}>
              Basado en investigación científica y modelos avanzados de predicción
            </Text>
          </VStack>
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10}>
            <Feature
              icon={FaChartPie}
              title={'Precisión y Confiabilidad'}
              text={'Algoritmos entrenados con miles de perfiles para asegurar una predicción certera de tu eneagrama.'}
            />
            <Feature
              icon={FaLightbulb}
              title={'Explicaciones Detalladas'}
              text={'Entendimiento profundo de tu tipo de personalidad y cómo influye en tus decisiones.'}
            />
            <Feature
              icon={FaUser}
              text={'Descubre cómo tu personalidad interactúa con otros tipos y mejora tus relaciones.'}
              title={'Autoconocimiento'}
            />
          </SimpleGrid>
        </Container>
      </Box>

      {/* Call to action */}
      <Box py={16}>
        <Container maxW={'7xl'}>
          <VStack spacing={2} textAlign={'center'}>
            <Heading as={'h2'} fontSize={'3xl'}>
              ¿Listo para descubrir tu tipo de eneagrama?
            </Heading>
            <Text color={'gray.500'} px={8} py={4} maxW={'2xl'}>
              Responde nuestro test de 80 preguntas y obtén resultados inmediatos sobre tu tipo de eneagrama y ala.
            </Text>
            <Button
              as={RouterLink}
              to='/test'
              rounded={'full'}
              size={'lg'}
              fontWeight={'normal'}
              px={6}
              mt={4}
              colorScheme={'blue'}
              bg={'primary.500'}
              _hover={{ bg: 'primary.600' }}>
              Comenzar Ahora
            </Button>
          </VStack>
        </Container>
      </Box>
    </Box>
  );
};

// Add missing Flex component
const Flex = Box;

export default Landing;
