import React from 'react';
import {
  Box,
  Container,
  Stack,
  Text,
  Link,
  useColorModeValue,
} from '@chakra-ui/react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <Box
      bg={useColorModeValue('gray.50', 'gray.900')}
      color={useColorModeValue('gray.700', 'gray.200')}
      borderTop={1}
      borderStyle={'solid'}
      borderColor={useColorModeValue('gray.200', 'gray.700')}
    >
      <Container
        as={Stack}
        maxW={'container.xl'}
        py={4}
        direction={{ base: 'column', md: 'row' }}
        spacing={4}
        justify={{ base: 'center', md: 'space-between' }}
        align={{ base: 'center', md: 'center' }}
      >
        <Text>© {currentYear} Enneagram Predictor. Todos los derechos reservados.</Text>
        <Stack direction={'row'} spacing={6}>
          <Link href={'#'}>Política de privacidad</Link>
          <Link href={'#'}>Términos de uso</Link>
          <Link href={'#'}>Contacto</Link>
        </Stack>
      </Container>
    </Box>
  );
};

export default Footer;
