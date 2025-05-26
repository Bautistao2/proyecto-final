import React from 'react';
import {
  Box,
  Flex,
  Text,
  Button,
  Stack,
  Link,
  Container,
  useColorModeValue,
  Heading,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const Header: React.FC = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box 
      as="header"
      bg={bgColor}
      borderBottom={1}
      borderStyle={'solid'}
      borderColor={borderColor}
      px={4}
      boxShadow={'sm'}
    >
      <Container maxW="container.xl">
        <Flex h={16} alignItems={'center'} justifyContent={'space-between'}>
          <Heading as={RouterLink} to="/" size="md" color="primary.500">
            Enneagram Predictor
          </Heading>

          <Flex alignItems={'center'}>
            <Stack direction={'row'} spacing={4}>
              <Button
                as={RouterLink}
                to="/"
                fontSize={'sm'}
                fontWeight={500}
                variant={'ghost'}
                colorScheme={'blue'}
              >
                Inicio
              </Button>
              
              <Button
                as={RouterLink}
                to="/test"
                fontSize={'sm'}
                fontWeight={600}
                variant={'solid'}
                colorScheme={'blue'}
              >
                Realizar Test
              </Button>
            </Stack>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
};

export default Header;
