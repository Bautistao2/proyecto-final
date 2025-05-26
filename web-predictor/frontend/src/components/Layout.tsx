import React from 'react';
import { 
  Box, 
  Flex, 
  Container,
  useColorModeValue
} from '@chakra-ui/react';
import Header from './Header';
import Footer from './Footer';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const bgColor = useColorModeValue('gray.50', 'gray.900');

  return (
    <Flex 
      direction="column" 
      minH="100vh"
      bg={bgColor}
    >
      <Header />
      <Box as="main" flex="1">
        <Container maxW="container.xl" py={8}>
          {children}
        </Container>
      </Box>
      <Footer />
    </Flex>
  );
};

export default Layout;
